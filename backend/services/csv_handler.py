from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import csv
import json
import os
import uuid
import shutil
from pathlib import Path


# Pydantic Models
class TransactionData(BaseModel):
    booking_date: str
    value_date: str
    status: str
    payer: str
    payee: str
    purpose: str
    transaction_type: str
    iban: str
    amount: float
    currency: str = "EUR"
    creditor_id: Optional[str] = None
    mandate_reference: Optional[str] = None
    customer_reference: Optional[str] = None
    category: Optional[str] = None
    confidence: Optional[str] = None
    reasoning: Optional[str] = None


class UploadSummary(BaseModel):
    total_transactions: int
    date_range: Dict[str, str]
    balance_total: float
    income_total: float
    expense_total: float
    transaction_types: Dict[str, int]


class UploadMetadata(BaseModel):
    upload_id: str
    format: str
    uploaded_at: str
    filename: str
    summary: UploadSummary


class UploadResponse(BaseModel):
    upload_id: str
    format: str
    uploaded_at: str
    filename: str
    summary: UploadSummary
    message: str


class UploadListResponse(BaseModel):
    uploads: List[UploadMetadata]
    total_count: int


class TransactionListResponse(BaseModel):
    upload_id: str
    transactions: List[TransactionData]
    pagination: Dict[str, int]


# Abstract Parser Base Class
class DataFormatParser(ABC):
    """Abstract base class for parsing different data formats"""

    @abstractmethod
    def parse(self, file_content: str) -> Dict[str, Any]:
        """Parse file content and return structured data"""
        pass

    @abstractmethod
    def validate_format(self, headers: List[str]) -> bool:
        """Validate if the file matches this format"""
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the format name"""
        pass


# German Bank CSV Parser Implementation
class GermanBankCSVParser(DataFormatParser):
    """Parser for German bank export CSV files"""

    EXPECTED_COLUMNS = [
        "Buchungsdatum",
        "Wertstellung",
        "Status",
        "Zahlungspflichtige*r",
        "Zahlungsempfänger*in",
        "Verwendungszweck",
        "Umsatztyp",
        "IBAN",
        "Betrag (€)",
        "Gläubiger-ID",
        "Mandatsreferenz",
        "Kundenreferenz"
    ]

    def get_format_name(self) -> str:
        return "german_bank"

    def validate_format(self, headers: List[str]) -> bool:
        """Check if headers match expected German bank format"""
        # Allow for slight variations in column names
        return all(col in headers for col in self.EXPECTED_COLUMNS[:9])  # At least first 9 columns

    def parse(self, file_content: str) -> Dict[str, Any]:
        """Parse German bank CSV file"""
        transactions = []
        lines = file_content.strip().split('\n')

        # Skip header rows (Girokonto, Zeitraum, Kontostand, empty line)
        data_start_idx = 0
        for i, line in enumerate(lines):
            if any(col in line for col in self.EXPECTED_COLUMNS[:3]):
                data_start_idx = i
                break

        print(f"[DEBUG] Starting to parse from line {data_start_idx}")
        print(f"[DEBUG] Total lines to process: {len(lines) - data_start_idx}")

        # Parse CSV data with semicolon delimiter (common in German CSV files)
        csv_data = '\n'.join(lines[data_start_idx:])
        reader = csv.DictReader(csv_data.split('\n'), delimiter=';', quotechar='"')

        row_number = data_start_idx + 1  # +1 because header is on data_start_idx
        skipped_count = 0
        empty_count = 0

        for row in reader:
            row_number += 1

            # Check if row is empty or invalid
            if not row or all(not v or not v.strip() for v in row.values()):
                empty_count += 1
                print(f"[DEBUG] Row {row_number}: Skipped (empty row)")
                continue

            try:
                # Parse amount (handle German number format: period as thousand separator, comma as decimal)
                amount_str = row.get("Betrag (€)", "0").strip()

                # Check if essential fields are present
                booking_date = row.get("Buchungsdatum", "").strip()
                if not booking_date:
                    skipped_count += 1
                    print(f"[DEBUG] Row {row_number}: Skipped (missing booking date). Row data: {row}")
                    continue

                # Remove currency symbol and spaces
                amount_str = amount_str.replace("€", "").replace(" ", "").strip()

                # Handle German number format:
                # 1. Remove thousand separators (periods)
                # 2. Replace decimal separator (comma) with period
                amount_str = amount_str.replace(".", "")  # Remove thousand separators
                amount_str = amount_str.replace(",", ".")  # Replace decimal separator

                try:
                    amount = float(amount_str) if amount_str else 0.0
                except ValueError:
                    skipped_count += 1
                    print(f"[DEBUG] Row {row_number}: Skipped (invalid amount '{amount_str}'). Row data: {row}")
                    continue

                transaction = TransactionData(
                    booking_date=booking_date,
                    value_date=row.get("Wertstellung", "").strip(),
                    status=row.get("Status", "").strip(),
                    payer=row.get("Zahlungspflichtige*r", "").strip(),
                    payee=row.get("Zahlungsempfänger*in", "").strip(),
                    purpose=row.get("Verwendungszweck", "").strip(),
                    transaction_type=row.get("Umsatztyp", "").strip(),
                    iban=row.get("IBAN", "").strip(),
                    amount=amount,
                    currency="EUR",
                    creditor_id=row.get("Gläubiger-ID", "").strip() or None,
                    mandate_reference=row.get("Mandatsreferenz", "").strip() or None,
                    customer_reference=row.get("Kundenreferenz", "").strip() or None
                )
                transactions.append(transaction)

                if len(transactions) <= 3 or len(transactions) % 50 == 0:
                    print(f"[DEBUG] Row {row_number}: Successfully parsed transaction (Total so far: {len(transactions)})")

            except Exception as e:
                # Skip invalid rows but continue parsing
                skipped_count += 1
                print(f"[DEBUG] Row {row_number}: Skipped due to error: {type(e).__name__}: {str(e)}")
                print(f"[DEBUG] Row {row_number}: Row data: {row}")
                continue

        print(f"[DEBUG] Parsing complete:")
        print(f"[DEBUG] - Total transactions parsed: {len(transactions)}")
        print(f"[DEBUG] - Empty rows skipped: {empty_count}")
        print(f"[DEBUG] - Invalid rows skipped: {skipped_count}")
        print(f"[DEBUG] - Total rows processed: {row_number - data_start_idx - 1}")

        return {"transactions": transactions}


# CSV Handler Service
class CSVHandlerService:
    """Service for handling CSV uploads and parsing"""

    def __init__(self):
        self.base_upload_dir = Path("/home/gregster/Documents/Projects/Finance_Dashboard/backend/data/uploads")
        self.temp_dir = Path("/home/gregster/Documents/Projects/Finance_Dashboard/backend/data/temp_uploads")
        self.parsers = [GermanBankCSVParser()]

        # Ensure directories exist
        self.base_upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def detect_format(self, file_content: str) -> Optional[DataFormatParser]:
        """Auto-detect file format based on headers"""
        lines = file_content.strip().split('\n')

        # Find header line - try both comma and semicolon delimiters
        for line in lines:
            # Try semicolon delimiter first (common in German CSV)
            try:
                headers = next(csv.reader([line], delimiter=';', quotechar='"'))
                for parser in self.parsers:
                    if parser.validate_format(headers):
                        return parser
            except:
                pass

            # Try comma delimiter
            try:
                headers = next(csv.reader([line], delimiter=','))
                for parser in self.parsers:
                    if parser.validate_format(headers):
                        return parser
            except:
                continue

        return None

    def calculate_summary(self, transactions: List[TransactionData]) -> UploadSummary:
        """Calculate summary statistics from transactions"""
        if not transactions:
            return UploadSummary(
                total_transactions=0,
                date_range={"from": "", "to": ""},
                balance_total=0.0,
                income_total=0.0,
                expense_total=0.0,
                transaction_types={}
            )

        # Calculate totals
        total_transactions = len(transactions)
        income_total = sum(t.amount for t in transactions if t.amount > 0)
        expense_total = abs(sum(t.amount for t in transactions if t.amount < 0))
        balance_total = sum(t.amount for t in transactions)

        # Transaction types breakdown
        transaction_types = {}
        for t in transactions:
            type_name = t.transaction_type or "UNKNOWN"
            transaction_types[type_name] = transaction_types.get(type_name, 0) + 1

        # Date range
        dates = [t.booking_date for t in transactions if t.booking_date]
        date_from = min(dates) if dates else ""
        date_to = max(dates) if dates else ""

        return UploadSummary(
            total_transactions=total_transactions,
            date_range={"from": date_from, "to": date_to},
            balance_total=round(balance_total, 2),
            income_total=round(income_total, 2),
            expense_total=round(expense_total, 2),
            transaction_types=transaction_types
        )

    def process_upload(self, file_content: str, filename: str) -> UploadResponse:
        """Process uploaded CSV file"""
        # Detect format
        parser = self.detect_format(file_content)
        if not parser:
            raise ValueError("Unsupported file format. Please upload a valid German bank CSV file.")

        # Parse file
        parsed_data = parser.parse(file_content)
        transactions = parsed_data.get("transactions", [])

        if not transactions:
            raise ValueError("No valid transactions found in the file.")

        # Generate unique ID
        upload_id = str(uuid.uuid4())
        uploaded_at = datetime.utcnow().isoformat() + "Z"

        # Calculate summary
        summary = self.calculate_summary(transactions)

        # Create upload directory
        upload_dir = self.base_upload_dir / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = UploadMetadata(
            upload_id=upload_id,
            format=parser.get_format_name(),
            uploaded_at=uploaded_at,
            filename=filename,
            summary=summary
        )

        with open(upload_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)

        # Save transactions
        transactions_data = {
            "transactions": [t.model_dump() for t in transactions]
        }

        with open(upload_dir / "transactions.json", "w", encoding="utf-8") as f:
            json.dump(transactions_data, f, indent=2, ensure_ascii=False)

        return UploadResponse(
            upload_id=upload_id,
            format=parser.get_format_name(),
            uploaded_at=uploaded_at,
            filename=filename,
            summary=summary,
            message="File uploaded and processed successfully"
        )

    def get_all_uploads(self) -> UploadListResponse:
        """Get list of all uploads"""
        uploads = []

        for upload_dir in self.base_upload_dir.iterdir():
            if upload_dir.is_dir():
                metadata_file = upload_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        uploads.append(UploadMetadata(**metadata))

        # Sort by uploaded_at descending (most recent first)
        uploads.sort(key=lambda x: x.uploaded_at, reverse=True)

        return UploadListResponse(
            uploads=uploads,
            total_count=len(uploads)
        )

    def get_upload_metadata(self, upload_id: str) -> UploadMetadata:
        """Get metadata for a specific upload"""
        upload_dir = self.base_upload_dir / upload_id
        metadata_file = upload_dir / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Upload {upload_id} not found")

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            return UploadMetadata(**metadata)

    def get_transactions(self, upload_id: str, offset: int = 0, limit: int = 50) -> TransactionListResponse:
        """Get transactions for a specific upload with pagination"""
        upload_dir = self.base_upload_dir / upload_id
        transactions_file = upload_dir / "transactions.json"

        if not transactions_file.exists():
            raise FileNotFoundError(f"Transactions for upload {upload_id} not found")

        with open(transactions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_transactions = [TransactionData(**t) for t in data.get("transactions", [])]

        # Apply pagination
        total = len(all_transactions)
        paginated_transactions = all_transactions[offset:offset + limit]

        return TransactionListResponse(
            upload_id=upload_id,
            transactions=paginated_transactions,
            pagination={
                "offset": offset,
                "limit": limit,
                "total": total
            }
        )

    def delete_upload(self, upload_id: str) -> bool:
        """Delete an upload and all its data"""
        upload_dir = self.base_upload_dir / upload_id

        if not upload_dir.exists():
            raise FileNotFoundError(f"Upload {upload_id} not found")

        # Remove directory and all contents
        shutil.rmtree(upload_dir)
        return True


# Singleton instance
_csv_handler_service = None


def get_csv_handler_service() -> CSVHandlerService:
    """Get singleton instance of CSV handler service"""
    global _csv_handler_service
    if _csv_handler_service is None:
        _csv_handler_service = CSVHandlerService()
    return _csv_handler_service
