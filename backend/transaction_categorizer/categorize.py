import json
import argparse
from datetime import datetime
from ollama import chat
from pydantic import BaseModel
from tqdm import tqdm

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
class CategorizedTransaction(BaseModel):
    category: str
    confidence: str  # "high", "medium", "low"
    reasoning: str


class CategorizedTransactionList(BaseModel):
    categorized: list[CategorizedTransaction]


# =============================================================================
# Prompt template
# =============================================================================
SYSTEM_PROMPT = """You are a financial transaction categorizer. Your task is to assign each transaction to exactly one category from the provided list.

Available categories:
{categories}

For each transaction, analyze:
- The payee/payer names
- The purpose/description
- The transaction type
- The amount

Respond with a category, confidence level (high/medium/low), and brief reasoning."""


def build_transaction_prompt(transaction: dict) -> str:
    """Format a single transaction for the prompt."""
    parts = [
        f"Date: {transaction.get('booking_date', 'N/A')}",
        f"Payer: {transaction.get('payer', 'N/A')}",
        f"Payee: {transaction.get('payee', 'N/A')}",
        f"Purpose: {transaction.get('purpose', 'N/A')}",
        f"Type: {transaction.get('transaction_type', 'N/A')}",
        f"Amount: {transaction.get('amount', 0)} {transaction.get('currency', 'EUR')}",
    ]
    return "\n".join(parts)


def categorize_transactions(
    transactions: list[dict],
    model: str = "llama3.3",
    batch_size: int = 5,
) -> list[dict]:
    """
    Categorize a list of transactions using the specified LLM.

    Args:
        transactions: List of transaction dictionaries
        model: Ollama model to use
        batch_size: Number of transactions to process per LLM call

    Returns:
        List of transactions with added 'category_info' field
    """
    results = []
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(categories="\n".join(f"- {c}" for c in CATEGORIES))
    }

    for i in tqdm(range(0, len(transactions), batch_size), desc="Categorizing"):
        batch = transactions[i:i + batch_size]

        # Build prompt for batch
        transaction_texts = []
        for idx, txn in enumerate(batch):
            transaction_texts.append(f"Transaction {idx + 1}:\n{build_transaction_prompt(txn)}")

        user_prompt = (
            "Categorize each of the following transactions:\n\n"
            + "\n\n---\n\n".join(transaction_texts)
        )

        messages = [
            system_message,
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = chat(
                model=model,
                messages=messages,
                format=CategorizedTransactionList.model_json_schema(),
                options={"temperature": 0},
            )
            categorized = CategorizedTransactionList.model_validate_json(
                response.message.content
            ).categorized

            # Match results back to transactions
            for txn, cat_info in zip(batch, categorized):
                txn_result = txn.copy()
                txn_result["category_info"] = {
                    "category": cat_info.category,
                    "confidence": cat_info.confidence,
                    "reasoning": cat_info.reasoning,
                }
                results.append(txn_result)

        except Exception as e:
            print(f"Error processing batch: {e}")
            # Add uncategorized entries on error
            for txn in batch:
                txn_result = txn.copy()
                txn_result["category_info"] = {
                    "category": "Miscellaneous",
                    "confidence": "low",
                    "reasoning": f"Error during categorization: {str(e)}",
                }
                results.append(txn_result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Categorize financial transactions using LLM")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to JSON file containing transactions"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: input_categorized.json)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3.3",
        help="Ollama model to use (default: llama3.3)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="Transactions per LLM call (default: 5)"
    )

    args = parser.parse_args()

    # Load transactions
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both {"transactions": [...]} and [...] formats
    if isinstance(data, dict) and "transactions" in data:
        transactions = data["transactions"]
    elif isinstance(data, list):
        transactions = data
    else:
        raise ValueError("Input must be a list or dict with 'transactions' key")

    print(f"Loaded {len(transactions)} transactions")
    print(f"Using model: {args.model}")

    # Categorize
    categorized = categorize_transactions(
        transactions,
        model=args.model,
        batch_size=args.batch_size,
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = args.input_file.rsplit(".", 1)[0]
        output_path = f"{base}_categorized.json"

    # Save results
    output_data = {"transactions": categorized}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved categorized transactions to {output_path}")

    # Print summary
    category_counts = {}
    for txn in categorized:
        cat = txn["category_info"]["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nCategory summary:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
