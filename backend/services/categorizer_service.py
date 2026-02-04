"""
Transaction categorization service using Ollama LLM.
Integrates with the existing CSV handler and provides progress tracking.
"""

import json
from typing import List, Dict, Optional, Callable
from pathlib import Path
from pydantic import BaseModel, field_validator
from ollama import chat


# =============================================================================
# CATEGORIES - Edit this list to customize your transaction categories
# =============================================================================
CATEGORIES = [
    "Groceries",
    "Dining & Restaurants",
    "Transportation",
    "Utilities",
    "Rent & Housing",
    "Insurance",
    "Healthcare",
    "Entertainment",
    "Shopping",
    "Subscriptions",
    "Income & Salary",
    "Transfer",
    "Fees & Charges",
    "Investment",
    "Cash Withdrawal",
    "Miscellaneous",
]


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
SYSTEM_PROMPT = """You are a financial transaction categorizer for German bank statements. Your task is to assign each transaction to exactly one category from the provided list.

Available categories:
{categories}

Important rules:
- "LOHN / GEHALT" or "Gehalt" in the purpose means salary → "Income & Salary"
- "Eingang" transaction type with employer name means salary → "Income & Salary"
- Bakeries (Bäckerei, Bäcker, Backwaren) → "Groceries"
- Supermarkets (REWE, ALDI, LIDL, EDEKA, Picnic, Netto) → "Groceries"
- "VISA Debitkartenumsatz" just means debit card payment, focus on the PAYEE name
- Subscriptions (Netflix, Spotify, Claude, OpenAI, Disney+, Amazon Prime) → "Subscriptions"
- "Miete" in purpose → "Rent & Housing"
- Restaurants, cafés, takeout → "Dining & Restaurants"
- PayPal purchases → look at the merchant name after "Ihr Einkauf bei"

For each transaction, analyze the payee name and purpose to determine the category. Focus on WHAT was purchased, not HOW it was paid."""


def build_transaction_prompt(transaction: dict) -> str:
    parts = [
        f"Date: {transaction.get('booking_date', 'N/A')}",
        f"Payer: {transaction.get('payer', 'N/A')}",
        f"Payee: {transaction.get('payee', 'N/A')}",
        f"Purpose: {transaction.get('purpose', 'N/A')}",
        f"Type: {transaction.get('transaction_type', 'N/A')}",
        f"Amount: {transaction.get('amount', 0)} {transaction.get('currency', 'EUR')}",
    ]
    return "\n".join(parts)


class CategorizerService:
    """Service for categorizing transactions using LLM."""

    def __init__(self, model: str = "gemma3:4b"):
        self.model = model
        self.categories = CATEGORIES

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
                categories="\n".join(f"- {c}" for c in self.categories)
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
