#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse

import feedparser
import httpx
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

import logging

logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to reduce noise
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Silence overly verbose loggers from libraries
for noisy in ["httpx", "httpcore", "asyncio", "urllib3", "requests"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("rss_semantic_monitor")

DB_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS feeds (
    url TEXT PRIMARY KEY,
    title TEXT,
    last_checked TEXT,
    ok INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_url TEXT,
    guid TEXT,
    link TEXT,
    title TEXT,
    published TEXT,
    summary TEXT,
    content_text TEXT,
    score REAL,
    topics_csv TEXT,
    hash TEXT UNIQUE,
    added_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(feed_url) REFERENCES feeds(url)
);
CREATE INDEX IF NOT EXISTS idx_entries_published ON entries(published);
CREATE INDEX IF NOT EXISTS idx_entries_score ON entries(score);
"""

@dataclasses.dataclass
class Topic:
    name: str
    prompts: List[str]

def load_feeds(path: str) -> List[str]:
    """Load and clean feed URLs from file"""
    feeds = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                
                # Extract URL if it has description in parentheses
                if " (" in line:
                    url = line.split(" (")[0].strip()
                else:
                    url = line.strip()
                
                # Validate URL format
                parsed = urlparse(url)
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    feeds.append(url)
                else:
                    logger.warning(f"Skipping invalid URL: {line}")
        
        logger.info(f"Loaded {len(feeds)} valid feed URLs")
        return feeds
    except FileNotFoundError:
        raise FileNotFoundError(f"Feeds file not found: {path}")

def load_topics(path: str) -> List[Topic]:
    """Load topics from YAML file with error handling"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data or "topics" not in data:
            raise ValueError(f"Invalid topics file format. Expected 'topics' key in {path}")
        
        topics = []
        for t in data["topics"]:
            if "name" not in t:
                logger.warning(f"Skipping topic without name: {t}")
                continue
            topics.append(Topic(t["name"], t.get("prompts", [t["name"]])))
        
        if not topics:
            raise ValueError("No valid topics found in file")
            
        logger.info(f"Loaded {len(topics)} topics")
        return topics
    except FileNotFoundError:
        raise FileNotFoundError(f"Topics file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in topics file: {e}")

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize database with error handling"""
    try:
        con = sqlite3.connect(db_path)
        con.execute("PRAGMA foreign_keys = ON")
        
        for stmt in DB_SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                con.execute(s)
        con.commit()
        return con
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def normalize_text(*parts: Optional[str]) -> str:
    """Normalize text with better error handling"""
    try:
        text = " ".join([str(p) if p is not None else "" for p in parts])
        text = re.sub(r"<[^>]+>", " ", text)  # strip HTML
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]  # Limit text length
    except Exception as e:
        logger.debug(f"Text normalization failed: {e}")
        return ""

def entry_datetime(entry: Dict[str, Any]) -> Optional[str]:
    """Parse entry datetime with better error handling"""
    for key in ("published", "updated", "created"):
        try:
            if key + "_parsed" in entry and entry[key + "_parsed"]:
                parsed_time = entry[key + "_parsed"]
                if len(parsed_time) >= 6:
                    return dt.datetime(*parsed_time[:6], tzinfo=dt.timezone.utc).isoformat()
            
            if key in entry and entry[key]:
                date_str = str(entry[key])
                # Handle Z suffix and other common formats
                date_str = date_str.replace("Z", "+00:00")
                try:
                    parsed_dt = dt.datetime.fromisoformat(date_str)
                    return parsed_dt.isoformat()
                except ValueError:
                    continue
        except Exception as e:
            logger.debug(f"Failed to parse datetime for key {key}: {e}")
            continue
    return None

async def fetch_feed(client, url, timeout_s=20.0):
    """Fetch feed with better error handling"""
    try:
        r = await client.get(url, timeout=timeout_s, follow_redirects=True)
        r.raise_for_status()
        
        logger.debug(f"Fetched {url} ({len(r.text)} bytes)")
        return url, r.text
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching {url}")
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP {e.response.status_code} for {url}")
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {type(e).__name__}: {e}")
    
    return url, None

async def gather_feeds(feed_urls: List[str], concurrency: int = 10) -> Dict[str, Optional[str]]:
    """Gather feeds with controlled concurrency"""
    if not feed_urls:
        logger.warning("No feed URLs provided")
        return {}
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def fetch_with_semaphore(client, url):
        async with semaphore:
            return await fetch_feed(client, url)
    
    timeout = httpx.Timeout(20.0, connect=10.0)
    
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": "rss-semantic-monitor/1.0 (RSS feed aggregator)",
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"
            }
        ) as client:
            tasks = [fetch_with_semaphore(client, url) for url in feed_urls]
            results = {}
            
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching feeds"):
                url, text = await coro
                results[url] = text
            
            successful = sum(1 for v in results.values() if v is not None)
            logger.info(f"Successfully fetched {successful}/{len(feed_urls)} feeds")
            return results
    except Exception as e:
        logger.error(f"Failed to gather feeds: {e}")
        return {}

def parse_entries(feed_text: str):
    """Parse entries with better error handling"""
    try:
        parsed = feedparser.parse(feed_text)
        
        if hasattr(parsed, 'bozo') and parsed.bozo:
            logger.debug(f"Feed parsing warning: {getattr(parsed, 'bozo_exception', 'Unknown error')}")
        
        feed_title = parsed.feed.get("title", "")
        logger.debug(f"Parsing feed '{feed_title}' with {len(parsed.entries)} entries")
        
        entries = []
        for e in parsed.entries:
            try:
                # Extract content safely
                content_parts = []
                if hasattr(e, 'content') and e.content:
                    for content_item in e.content:
                        content_parts.append(content_item.get("value", ""))
                
                # Extract tags safely  
                tag_parts = []
                if hasattr(e, 'tags') and e.tags:
                    for tag in e.tags:
                        tag_parts.append(tag.get("term", ""))
                
                entry_data = {
                    "guid": e.get("id") or e.get("guid") or e.get("link") or "",
                    "link": e.get("link", ""),
                    "title": e.get("title", ""),
                    "summary": e.get("summary", ""),
                    "content_text": normalize_text(
                        e.get("title", ""),
                        e.get("summary", ""),
                        " ".join(content_parts),
                        " ".join(tag_parts)
                    ),
                    "published": entry_datetime(e)
                }
                
                # Skip entries without meaningful content
                if not entry_data["content_text"].strip():
                    continue
                    
                entries.append(entry_data)
            except Exception as ex:
                logger.debug(f"Failed to parse entry: {ex}")
                continue
        
        return feed_title, entries
    except Exception as e:
        logger.error(f"Failed to parse feed: {e}")
        return "", []

def upsert_feed(con: sqlite3.Connection, url: str, title: str, ok: int = 1):
    """Upsert feed with error handling"""
    try:
        con.execute(
            "INSERT INTO feeds(url, title, last_checked, ok) VALUES(?,?,datetime('now'),?) "
            "ON CONFLICT(url) DO UPDATE SET title=excluded.title, last_checked=excluded.last_checked, ok=excluded.ok",
            (url, title[:500], ok),
        )
    except sqlite3.Error as e:
        logger.error(f"Failed to upsert feed {url}: {e}")

def insert_entry(con: sqlite3.Connection, feed_url: str, entry: Dict[str, Any], score: float, topics: List[str]):
    """Insert entry with better error handling"""
    try:
        hash_input = (entry.get("guid", "") + entry.get("link", "") + entry.get("title", ""))[:1000]
        h = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        
        with contextlib.suppress(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO entries(feed_url, guid, link, title, published, summary, content_text, score, topics_csv, hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    feed_url,
                    entry.get("guid", "")[:500],
                    entry.get("link", "")[:500],
                    entry.get("title", "")[:500],
                    entry.get("published"),
                    entry.get("summary", "")[:1000],
                    entry.get("content_text", "")[:5000],
                    float(score),
                    ",".join(topics)[:500],
                    h,
                )
            )
    except Exception as e:
        logger.error(f"Failed to insert entry: {e}")

def encode_topics(model: SentenceTransformer, topics: List[Topic]):
    """Encode topics for similarity matching"""
    topic_names = [t.name for t in topics]
    topic_prompts = [p for t in topics for p in t.prompts]
    topic_prompt_index = []
    
    for i, t in enumerate(topics):
        topic_prompt_index.extend([i] * len(t.prompts))
    
    logger.info(f"Encoding {len(topic_prompts)} topic prompts...")
    emb = model.encode(topic_prompts, convert_to_tensor=True, normalize_embeddings=True)
    return topic_names, emb, topic_prompt_index

def classify_text(emb_text, emb_topic_prompts, topic_prompt_index, topic_names, threshold: float) -> Tuple[float, List[str]]:
    """Classify text against topics"""
    sims = util.cos_sim(emb_text, emb_topic_prompts)[0].cpu().numpy()
    max_per_topic = {}
    
    for sim, idx in zip(sims, topic_prompt_index):
        name = topic_names[idx]
        max_per_topic[name] = max(max_per_topic.get(name, -1.0), float(sim))
    
    matched = [name for name, s in max_per_topic.items() if s >= threshold]
    top_score = max(max_per_topic.values()) if max_per_topic else -1.0
    return float(top_score), matched

async def cmd_run(args):
    con = init_db(args.db)
    
    # Load feeds and topics
    feed_urls = load_feeds(args.feeds)
    topics = load_topics(args.topics)
    
    if not feed_urls:
        logger.error("No valid feed URLs found")
        return
    
    if not topics:
        logger.error("No valid topics found")
        return
    
    # Initialize model
    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    topic_names, emb_topic_prompts, topic_prompt_index = encode_topics(model, topics)

    # Fetch feeds
    results = await gather_feeds(feed_urls, concurrency=args.concurrency)
    
    # Track working vs broken feeds
    working_feeds = []
    broken_feeds = []
    
    for url in feed_urls:
        if results.get(url) is not None:
            working_feeds.append(url)
        else:
            broken_feeds.append(url)

    texts, meta = [], []
    for url, text in results.items():
        if not text:
            upsert_feed(con, url, title="", ok=0)
            continue
            
        feed_title, entries = parse_entries(text)
        logger.debug(f"Feed '{feed_title}' ({url}): {len(entries)} entries")
        upsert_feed(con, url, title=feed_title or url, ok=1)
        
        for e in entries[: args.max_items_per_feed or len(entries)]:
            # Filter by date if specified
            if args.since_days and e.get("published"):
                try:
                    pub_dt = dt.datetime.fromisoformat(e["published"].replace("Z", "+00:00"))
                    if pub_dt < dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.since_days):
                        continue
                except Exception:
                    pass  # Include entry if date parsing fails

            if e["content_text"].strip():  # Only include entries with content
                texts.append(e["content_text"])
                meta.append((url, e))

    if not texts:
        logger.warning("No entries to classify.")
        return
    
    logger.info(f"Classifying {len(texts)} entries...")

    # Process in batches
    all_scores_topics = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Processing batches"):
        batch_texts = texts[i : i + args.batch_size]
        emb_texts = model.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)
        
        for j in range(emb_texts.shape[0]):
            score, matched = classify_text(
                emb_texts[j:j+1], emb_topic_prompts, topic_prompt_index, topic_names, args.threshold
            )
            all_scores_topics.append((score, matched))

    # Store results
    stored_count = 0
    with con:
        for (url, e), (score, matched) in zip(meta, all_scores_topics):
            if matched:
                insert_entry(con, url, e, score, matched)
                stored_count += 1

    total_entries = con.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
    logger.info(f"Stored {stored_count} new matching entries. Total entries in DB: {total_entries}")

    if args.export_json:
        export_entries(con, args.export_json, limit=args.export_limit)
        logger.info(f"Exported to {args.export_json}")
    
    # Auto-cleanup broken feeds if requested
    if args.auto_cleanup and broken_feeds:
        if args.backup_before_cleanup:
            create_backup_and_cleanup(args.feeds, broken_feeds)
        else:
            cleanup_feeds_file(args.feeds, broken_feeds)
        logger.info(f"Updated {args.feeds} - marked {len(broken_feeds)} broken URLs as comments")
    elif broken_feeds:
        logger.info(f"Found {len(broken_feeds)} broken feeds. Use --auto-cleanup to remove them automatically")

def export_entries(con: sqlite3.Connection, path: str, limit: Optional[int] = None):
    """Export entries to JSON"""
    q = "SELECT title, link, published, score, topics_csv, feed_url FROM entries ORDER BY published DESC, score DESC"
    if limit:
        q += f" LIMIT {limit}"
    
    rows = con.execute(q).fetchall()
    data = [
        {
            "title": r[0], 
            "link": r[1], 
            "published": r[2], 
            "score": r[3],
            "topics": r[4].split(",") if r[4] else [], 
            "feed_url": r[5]
        }
        for r in rows
    ]
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def cleanup_feeds_file(feeds_file: str, broken_urls: List[str]):
    """Remove broken URLs from feeds file while preserving comments and structure"""
    if not broken_urls:
        return
        
    broken_set = set(broken_urls)
    
    # Read original file
    with open(feeds_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Filter out broken URLs while preserving comments and structure
    cleaned_lines = []
    removed_count = 0
    
    for line in lines:
        original_line = line
        stripped = line.strip()
        
        # Keep empty lines and comments
        if not stripped or stripped.startswith('#'):
            cleaned_lines.append(original_line)
            continue
            
        # Extract URL from line (handle descriptions in parentheses)
        url = stripped.split(' (')[0].strip() if ' (' in stripped else stripped
        
        # Check if this URL is broken
        if url in broken_set:
            # Comment out the broken URL instead of removing completely
            cleaned_lines.append(f"# BROKEN: {original_line}")
            removed_count += 1
            logger.debug(f"Marked as broken: {url}")
        else:
            cleaned_lines.append(original_line)
    
    # Write cleaned file back
    with open(feeds_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
        
    logger.info(f"Marked {removed_count} broken URLs as comments in {feeds_file}")

def create_backup_and_cleanup(feeds_file: str, broken_urls: List[str]):
    """Create backup and clean feeds file"""
    if not broken_urls:
        return
        
    # Create backup
    backup_file = f"{feeds_file}.backup_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    import shutil
    shutil.copy2(feeds_file, backup_file)
    logger.info(f"Created backup: {backup_file}")
    
    # Clean the file
    cleanup_feeds_file(feeds_file, broken_urls)

async def cmd_discover(args):
    """Discover RSS feeds from domains"""
    domains = []
    if args.domains_file:
        with open(args.domains_file, "r", encoding="utf-8") as f:
            domains.extend([ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")])
    if args.domains:
        domains.extend(args.domains)

    discovered = []
    timeout = httpx.Timeout(15.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for d in domains:
            try:
                if not d.startswith("http"):
                    url = "https://" + d
                    if d.count(".") == 1:
                        url = "https://www." + d
                else:
                    url = d
                
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                
                soup = BeautifulSoup(resp.text, "html.parser")
                for link in soup.find_all("link", rel="alternate"):
                    if link.get("type") in ["application/rss+xml", "application/atom+xml"]:
                        href = link.get("href")
                        if href:
                            if href.startswith("/"):
                                href = urljoin(url, href)
                            discovered.append(href)
            except Exception as e:
                logger.warning(f"Failed to discover feeds from {d}: {e}")
                continue

    unique = sorted(set(discovered))
    mode = "a" if os.path.exists(args.out) else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        for u in unique:
            f.write(u + "\n")

    logger.info(f"Discovered {len(unique)} feeds. Written to {args.out}")

async def cmd_clean(args):
    """Clean feeds file by testing all URLs and removing broken ones"""
    feed_urls = load_feeds(args.feeds)
    
    if not feed_urls:
        logger.error("No valid feed URLs found")
        return
    
    logger.info(f"Testing {len(feed_urls)} feed URLs...")
    results = await gather_feeds(feed_urls, concurrency=args.concurrency)
    
    broken_feeds = [url for url in feed_urls if results.get(url) is None]
    working_feeds = [url for url in feed_urls if results.get(url) is not None]
    
    logger.info(f"Results: {len(working_feeds)} working, {len(broken_feeds)} broken")
    
    if broken_feeds:
        if args.backup:
            create_backup_and_cleanup(args.feeds, broken_feeds)
        else:
            cleanup_feeds_file(args.feeds, broken_feeds)
        logger.info(f"Cleaned {args.feeds} - marked {len(broken_feeds)} broken URLs as comments")
    else:
        logger.info("All feeds are working - no cleanup needed")

async def cmd_export(args):
    """Export entries to JSON"""
    con = init_db(args.db)
    export_entries(con, args.out, limit=args.limit)
    logger.info(f"Exported to {args.out}")

def build_argparser():
    p = argparse.ArgumentParser(description="Semantic RSS monitor: discovery → fetch → classify → store")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run", help="Fetch feeds, classify, store to SQLite")
    pr.add_argument("--feeds", required=True, help="File containing RSS feed URLs")
    pr.add_argument("--topics", required=True, help="YAML file containing topics and prompts")
    pr.add_argument("--db", default="rss_semantic.db", help="SQLite database path")
    pr.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence transformer model")
    pr.add_argument("--threshold", type=float, default=0.40, help="Similarity threshold for topic matching")
    pr.add_argument("--batch-size", type=int, default=32, help="Batch size for text encoding")
    pr.add_argument("--concurrency", type=int, default=10, help="Concurrent feed fetches")
    pr.add_argument("--max-items-per-feed", type=int, default=50, help="Max items to process per feed")
    pr.add_argument("--since-days", type=int, default=7, help="Only process entries from last N days")
    pr.add_argument("--export-json", help="Export results to JSON file")
    pr.add_argument("--export-limit", type=int, default=500, help="Limit exported entries")
    pr.add_argument("--auto-cleanup", action="store_true", help="Automatically remove broken feed URLs from feeds file")
    pr.add_argument("--backup-before-cleanup", action="store_true", help="Create backup before cleaning feeds file")

    # clean
    pc = sub.add_parser("clean", help="Test all feeds and mark broken ones as comments")
    pc.add_argument("--feeds", required=True, help="File containing RSS feed URLs")
    pc.add_argument("--concurrency", type=int, default=10, help="Concurrent feed fetches")
    pc.add_argument("--backup", action="store_true", help="Create backup before cleaning")

    # discover
    pd = sub.add_parser("discover", help="Discover RSS feeds from domains/URLs")
    pd.add_argument("--domains-file", help="File containing domains to scan")
    pd.add_argument("--domains", nargs="*", help="Domains to scan for feeds")
    pd.add_argument("--out", default="feeds.txt", help="Output file for discovered feeds")

    # export
    pe = sub.add_parser("export", help="Export stored matched entries to JSON")
    pe.add_argument("--db", default="rss_semantic.db", help="SQLite database path")
    pe.add_argument("--out", required=True, help="Output JSON file")
    pe.add_argument("--limit", type=int, help="Limit number of exported entries")

    return p

def main(argv=None):
    argv = argv or sys.argv[1:]
    p = build_argparser()
    args = p.parse_args(argv)

    try:
        if args.cmd == "run":
            asyncio.run(cmd_run(args))
        elif args.cmd == "clean":
            asyncio.run(cmd_clean(args))
        elif args.cmd == "discover":
            asyncio.run(cmd_discover(args))
        elif args.cmd == "export":
            asyncio.run(cmd_export(args))
        else:
            p.print_help()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()