"""
Transaction categorization service using Ollama LLM.
Integrates with the existing CSV handler and provides progress tracking.
"""

import json
import re
from typing import List, Dict, Optional, Callable
from pathlib import Path
from pydantic import BaseModel, field_validator
from ollama import chat


# =============================================================================
# CATEGORIES with German keywords for better LLM understanding
# =============================================================================
CATEGORIES = [
    "Groceries",
    "Dining & Restaurants",
    "Transportation",
    "Internet & Mobile",
    "Rent & Housing",
    "Insurance",
    "Healthcare",
    "Sport & Leisure",
    "Shopping",
    "Subscriptions",
    "Income & Salary",
    "Transfer",
    "Investment",
    "Cash Withdrawal",
    "Miscellaneous",
]

# German keywords mapped to each category — used in the prompt
CATEGORY_HINTS: dict[str, str] = {
    "Groceries": "Lebensmittel, Supermarkt, Bäckerei, Bäcker, REWE, ALDI, LIDL, EDEKA, Netto, Picnic, Penny, dm, Rossmann, Vollwert, Backwaren, Metzgerei",
    "Dining & Restaurants": "Restaurant, Gaststätte, Imbiss, Café, Kaffee, Pizza, Burger, Döner, Lieferando, McDonald, Starbucks",
    "Transportation": "Tankstelle, Benzin, DB Bahn, Deutsche Bahn, MVV, ADAC, Fahrkarte, Taxi, Uber, Flixbus, Shell, Aral, Jet, Parken, AVG, Augsburger Verkehrsgesellschaft, SWABI, swa, Mobilitäts-Abo",
    "Internet & Mobile": "Klarmobil, Telekom, Vodafone, O2, Mobilfunk, Handyvertrag, Internet, DSL, Glasfaser, Telefon",
    "Rent & Housing": "Miete, Wohnung, Nebenkosten, Hausgeld, Mietvertrag, Kaution, Strom, Gas, Wasser, Stadtwerke",
    "Insurance": "Versicherung, Krankenversicherung, Haftpflicht, KFZ-Versicherung, Allianz, HUK, ERGO, AOK, TK, Barmer",
    "Healthcare": "Arzt, Apotheke, Zahnarzt, Krankenhaus, Praxis, Medikament, Rezept, Physiotherapie",
    "Sport & Leisure": "Schwimmbad, Familienbad, Plärre, Sport Sheds, Fitnessstudio, Sportstudio, Kino, Theater, Konzert, Freizeit, Verein, Bowling, Kletterhalle",
    "Shopping": "Amazon, Zalando, eBay, Online-Shop, Kleidung, Elektronik, Möbel, IKEA, MediaMarkt, Saturn",
    "Subscriptions": "Abo, Abonnement, Netflix, Spotify, Disney+, Claude, OpenAI, Apple, Google, YouTube Premium, Amazon Prime, Anthropic, Rundfunk, GEZ, Beitragsservice",
    "Income & Salary": "Gehalt, Lohn, Einkommen, Bezüge, Arbeitgeber, LOHN / GEHALT, Gutschrift",
    "Transfer": "Überweisung, Umbuchung, Dauerauftrag, Kontowechsel, eigenes Konto",
    "Investment": "Depot, Aktie, Fonds, ETF, Sparplan, Wertpapier, Trade Republic, Scalable",
    "Cash Withdrawal": "Bargeld, Geldautomat, Abhebung, ATM, Auszahlung",
    "Miscellaneous": "Sonstige, Gebühr, Entgelt, Kontoführung",
}


# =============================================================================
# Input pre-processing — clean up noisy bank data before sending to LLM
# =============================================================================
def clean_transaction_for_prompt(transaction: dict) -> dict:
    """Strip noise from transaction data so the LLM sees cleaner input."""
    cleaned = transaction.copy()

    # Strip "VISA Debitkartenumsatz vom DD.MM.YYYY" from purpose — it's just noise
    purpose = cleaned.get("purpose", "") or ""
    purpose = re.sub(r"VISA Debitkartenumsatz vom \d{2}\.\d{2}\.\d{4}", "", purpose).strip()
    cleaned["purpose"] = purpose if purpose else ""

    # Clean payee: remove trailing city in format "/CITY"
    payee = cleaned.get("payee", "") or ""
    # "VOLLWERTBAECKEREI.SCHN/AUGSBURG" → "VOLLWERTBAECKEREI.SCHN" (keep merchant, drop city)
    # But preserve the original for context
    cleaned["payee_clean"] = re.sub(r"/[A-ZÄÖÜ]{2,}$", "", payee).strip()

    # Extract merchant from PayPal purpose
    if "PayPal" in (cleaned.get("payer", "") or "") or "PayPal" in payee:
        match = re.search(r"Ihr Einkauf bei (.+?)(?:,|$)", purpose)
        if match:
            cleaned["paypal_merchant"] = match.group(1).strip()

    return cleaned


# =============================================================================
# Pydantic models for structured LLM output
# =============================================================================
class SimpleTransaction(BaseModel):
    category: str

    @field_validator("category")
    @classmethod
    def must_be_valid_category(cls, v: str) -> str:
        if v not in CATEGORIES:
            return "Miscellaneous"
        return v


class SimpleTransactionList(BaseModel):
    categorized: list[SimpleTransaction]


class VerboseTransaction(BaseModel):
    category: str
    confidence: str
    reasoning: str

    @field_validator("category")
    @classmethod
    def must_be_valid_category(cls, v: str) -> str:
        if v not in CATEGORIES:
            return "Miscellaneous"
        return v


class VerboseTransactionList(BaseModel):
    categorized: list[VerboseTransaction]


# =============================================================================
# JSON schemas for ollama (enforces enum constraints on the model side)
# =============================================================================
def build_ollama_schema(verbose: bool = False) -> dict:
    category_field = {"type": "string", "enum": CATEGORIES}
    if verbose:
        item = {
            "type": "object",
            "properties": {
                "category": category_field,
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "reasoning": {"type": "string"},
            },
            "required": ["category", "confidence", "reasoning"],
        }
    else:
        item = {
            "type": "object",
            "properties": {"category": category_field},
            "required": ["category"],
        }
    return {
        "type": "object",
        "properties": {
            "categorized": {"type": "array", "items": item}
        },
        "required": ["categorized"],
    }


# =============================================================================
# Prompt template
# =============================================================================
SYSTEM_PROMPT = """You are a financial transaction categorizer for German bank statements (DKB, Sparkasse, etc.).
Assign each transaction to exactly ONE category.

Categories (with German keywords):
{categories}

## Few-shot examples

Transaction: Payee="REWE SAGT DANKE/AUGSBURG", Purpose="", Type="Ausgang", Amount=-42.15 EUR
→ Groceries

Transaction: Payee="AWI SOLUTIONS GMBH", Purpose="LOHN / GEHALT 01/26", Type="Eingang", Amount=2822.18 EUR
→ Income & Salary

Transaction: Payee="CLAUDE.AI.SUBSCRIPTION/ANTHROPIC.COM", Purpose="", Type="Ausgang", Amount=-21.42 EUR
→ Subscriptions

Transaction: Payee="Steib Claudia", Purpose="Miete Wohnung", Type="Ausgang", Amount=-400.00 EUR
→ Rent & Housing

Transaction: Payee="VOLLWERTBAECKEREI.SCHN", Purpose="", Type="Ausgang", Amount=-4.40 EUR
→ Groceries

Transaction: Payee="Picnic", Purpose="", Type="Ausgang", Amount=-31.00 EUR
→ Groceries

Transaction: Payee="PayPal", Purpose="Ihr Einkauf bei French Albion", Type="Ausgang", Amount=-119.90 EUR
→ Shopping

Transaction: Payee="Markus Siegfried Böhm", Purpose="", Type="Ausgang", Amount=-58.00 EUR
→ Transfer

Transaction: Payee="Klarmobil GmbH", Purpose="", Type="Ausgang", Amount=-9.99 EUR
→ Internet & Mobile

Transaction: Payee="Telekom Deutschland GmbH", Purpose="", Type="Ausgang", Amount=-39.95 EUR
→ Internet & Mobile

Transaction: Payee="AVG Augsburger Verkehrsgesellschaft mbH", Purpose="Mobilitäts-Abo", Type="Ausgang", Amount=-59.00 EUR
→ Transportation

Transaction: Payee="SWABI", Purpose="", Type="Ausgang", Amount=-5.00 EUR
→ Transportation

Transaction: Payee="FAMILIENBAD.AM.PLAERRE", Purpose="", Type="Ausgang", Amount=-4.50 EUR
→ Sport & Leisure

Transaction: Payee="SPORT SHEDS AUGSBURG", Purpose="", Type="Ausgang", Amount=-15.00 EUR
→ Sport & Leisure

Transaction: Payee="ARD ZDF Deutschlandradio", Purpose="Rundfunkbeitrag", Type="Ausgang", Amount=-55.08 EUR
→ Subscriptions

## Rules
- "VISA Debitkartenumsatz" just means debit card — ignore it, focus on the PAYEE
- Bakeries (Bäckerei, Bäcker, Vollwert, Backwaren) → Groceries, NOT Dining
- "Eingang" with "LOHN/GEHALT" or employer name → Income & Salary
- PayPal → look at merchant name after "Ihr Einkauf bei"
- "Miete" in purpose → Rent & Housing
- .AI., .COM subscriptions, streaming services, Rundfunk/GEZ → Subscriptions
- Picnic, REWE, ALDI, LIDL, EDEKA → Groceries
- Klarmobil, Telekom → Internet & Mobile (NOT Subscriptions)
- AVG, Augsburger Verkehrsgesellschaft, SWABI → Transportation
- Familienbad, Sport Sheds, Fitnessstudio → Sport & Leisure
- Strom, Gas, Wasser, Stadtwerke → Rent & Housing (part of living costs)
- Personal names with no clear purpose → Transfer"""


def build_transaction_prompt(transaction: dict) -> str:
    """Format a cleaned transaction for the prompt."""
    cleaned = clean_transaction_for_prompt(transaction)

    payee = cleaned.get("payee_clean", cleaned.get("payee", "N/A"))
    purpose = cleaned.get("purpose", "")
    paypal_merchant = cleaned.get("paypal_merchant", "")

    parts = [
        f"Payee: {payee}",
    ]

    # Only include purpose if it adds information
    if purpose:
        parts.append(f"Purpose: {purpose}")

    if paypal_merchant:
        parts.append(f"Merchant: {paypal_merchant}")

    parts.append(f"Type: {cleaned.get('transaction_type', 'N/A')}")
    parts.append(f"Amount: {cleaned.get('amount', 0)} EUR")

    return "\n".join(parts)


class CategorizerService:
    """Service for categorizing transactions using LLM."""

    def __init__(self, model: str = "gemma3:4b"):
        self.model = model
        self.categories = CATEGORIES

    def _build_category_list(self) -> str:
        """Build category list with German keyword hints."""
        lines = []
        for cat in self.categories:
            hints = CATEGORY_HINTS.get(cat, "")
            if hints:
                lines.append(f"- {cat} ({hints})")
            else:
                lines.append(f"- {cat}")
        return "\n".join(lines)

    def categorize_transactions(
        self,
        transactions: List[Dict],
        batch_size: int = 5,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict]:
        results = []
        total = len(transactions)

        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                categories=self._build_category_list()
            )
        }
        schema = build_ollama_schema(verbose)
        response_model = VerboseTransactionList if verbose else SimpleTransactionList

        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            current = i + len(batch)

            if progress_callback:
                progress_callback(
                    current,
                    total,
                    f"Categorizing transactions {i+1}-{current} of {total}..."
                )

            transaction_texts = []
            for idx, txn in enumerate(batch):
                transaction_texts.append(
                    f"Transaction {idx + 1}:\n{build_transaction_prompt(txn)}"
                )

            user_prompt = (
                "Categorize each of the following transactions:\n\n"
                + "\n\n---\n\n".join(transaction_texts)
            )

            messages = [system_message, {"role": "user", "content": user_prompt}]

            try:
                response = chat(
                    model=self.model,
                    messages=messages,
                    format=schema,
                    options={"temperature": 0},
                )
                categorized = response_model.model_validate_json(
                    response.message.content
                ).categorized

                for txn, cat_info in zip(batch, categorized):
                    txn_result = txn.copy()
                    txn_result["category"] = cat_info.category
                    if verbose:
                        txn_result["confidence"] = cat_info.confidence
                        txn_result["reasoning"] = cat_info.reasoning
                    results.append(txn_result)

            except Exception as e:
                print(f"Error processing batch: {e}")
                for txn in batch:
                    txn_result = txn.copy()
                    txn_result["category"] = "Miscellaneous"
                    if verbose:
                        txn_result["confidence"] = "low"
                        txn_result["reasoning"] = f"Error: {e}"
                    results.append(txn_result)

        if progress_callback:
            progress_callback(total, total, "Categorization complete!")

        return results

    def categorize_upload(
        self,
        upload_dir: Path,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict:
        transactions_path = upload_dir / "transactions.json"

        if not transactions_path.exists():
            raise FileNotFoundError(f"Transactions file not found: {transactions_path}")

        with open(transactions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        transactions = data.get("transactions", [])

        if not transactions:
            raise ValueError("No transactions found in file")

        categorized = self.categorize_transactions(
            transactions,
            batch_size=5,
            verbose=verbose,
            progress_callback=progress_callback,
        )

        # Save categorized transactions
        output_data = {"transactions": categorized}
        with open(transactions_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Calculate statistics
        category_counts: dict[str, int] = {}
        for txn in categorized:
            cat = txn["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_categorized": len(categorized),
            "category_counts": category_counts,
            "categories": sorted(category_counts.keys()),
        }


# Singleton instance
_categorizer_service: Optional[CategorizerService] = None


def get_categorizer_service(model: str = "gemma3:4b") -> CategorizerService:
    global _categorizer_service
    if _categorizer_service is None or _categorizer_service.model != model:
        _categorizer_service = CategorizerService(model=model)
    return _categorizer_service
