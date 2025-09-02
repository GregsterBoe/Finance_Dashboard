@echo off
REM ===============================
REM Semantic RSS Pipeline (Windows)
REM ===============================


REM 4) (Optional) Discover feeds from domains.txt into feeds.txt
REM if exist domains.txt (
  REM echo Discovering feeds from domains.txt...
  REM python rss_semantic_monitor.py discover --domains-file domains.txt --out feeds.txt
REM )

REM 5) Run the semantic monitor
echo Running semantic RSS monitor...
python rss_semantic_monitor.py run ^
  --feeds feeds.txt ^
  --topics topics.example.yaml ^
  --db rss_semantic.db ^
  --threshold 0.40 ^
  --export-json matched.json ^
  --auto-cleanup

REM 6) Export (redundant if already exported in run)
echo Exporting latest results...
python rss_semantic_monitor.py export --db rss_semantic.db --out matched_export.json --limit 200

echo Done! Results are in matched.json and matched_export.json
pause
