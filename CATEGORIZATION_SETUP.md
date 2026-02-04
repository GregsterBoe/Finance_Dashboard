# Transaction Categorization Feature - Setup Guide

This guide explains how to set up and use the AI-powered transaction categorization feature.

## Prerequisites

### 1. Install Ollama

Ollama is required to run the AI model locally.

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Or download from: https://ollama.com/download
```

### 2. Pull the Gemma2 Model

```bash
ollama pull gemma2:2b
```

This downloads the Gemma 2B model (~1.6GB). You can use other models by changing the model name in the code.

### 3. Install Python Dependencies

```bash
cd backend
pip install -r requirements_categorizer.txt
```

Required packages:
- `ollama>=0.1.0` - Ollama Python SDK
- `tqdm>=4.66.0` - Progress bars (optional, for standalone use)

## Architecture

### Backend Components

1. **`services/categorizer_service.py`**
   - Core categorization logic
   - Uses Ollama API to categorize transactions
   - Processes transactions in batches (default: 5 at a time)
   - Provides progress callbacks

2. **`services/categorizer_router.py`**
   - FastAPI endpoints for categorization
   - Streaming endpoint with Server-Sent Events (SSE) for real-time progress
   - Non-streaming endpoint for simple requests

3. **`backend/app.py`** (modified)
   - Registers categorization router at `/api/categorize/*`

### Frontend Components

1. **`components/CategorizationModal.tsx`**
   - Modal dialog for categorization
   - Real-time progress bar using SSE
   - Displays results and category breakdown

2. **`components/TransactionTable.tsx`** (modified)
   - Added "Category" column
   - Shows category badges with confidence indicators
   - Hover tooltip shows reasoning

3. **`pages/PersonalFinance.tsx`** (modified)
   - Added "Categorize" button to upload detail view
   - Integrates categorization modal

## API Endpoints

### Stream Categorization (with progress)
```
POST /api/categorize/stream/{upload_id}?model=gemma2:2b
```

Returns Server-Sent Events:
```json
data: {"type": "progress", "current": 10, "total": 50, "status": "...", "percentage": 20.0}
data: {"type": "complete", "total_categorized": 50, "category_counts": {...}}
data: {"type": "error", "error": "..."}
```

### Simple Categorization (no progress)
```
POST /api/categorize/{upload_id}?model=gemma2:2b
```

Returns:
```json
{
  "upload_id": "...",
  "total_categorized": 50,
  "category_counts": {
    "Groceries": 15,
    "Dining & Restaurants": 8,
    ...
  },
  "categories": ["Groceries", "Dining & Restaurants", ...]
}
```

### Get Available Categories
```
GET /api/categories
```

## Usage

### From the UI

1. Navigate to **Personal Finance** page
2. Click on an upload to view details
3. Click the **🏷️ Categorize** button
4. Watch real-time progress as transactions are categorized
5. View results showing category breakdown
6. Categories will appear in the transaction table

### From the Command Line

You can still use the standalone script:

```bash
cd backend
python transaction_categorizer/categorize.py data/uploads/YOUR_UPLOAD_ID/transactions.json
```

## Categories

The system uses these predefined categories:

- **Groceries** - Supermarket purchases, food shopping
- **Dining & Restaurants** - Eating out, takeout
- **Transportation** - Gas, public transport, ride-sharing
- **Utilities** - Electricity, water, internet, phone
- **Rent & Housing** - Rent payments, HOA fees
- **Insurance** - Health, car, home insurance
- **Healthcare** - Doctor visits, pharmacy, medical
- **Entertainment** - Movies, concerts, hobbies
- **Shopping** - Clothing, electronics, general retail
- **Subscriptions** - Netflix, Spotify, gym memberships
- **Income & Salary** - Paychecks, freelance income
- **Transfer** - Money transfers between accounts
- **Fees & Charges** - Bank fees, ATM charges
- **Investment** - Stock purchases, crypto, savings
- **Cash Withdrawal** - ATM withdrawals
- **Miscellaneous** - Everything else

### Customizing Categories

Edit `backend/services/categorizer_service.py`:

```python
CATEGORIES = [
    "Your Custom Category 1",
    "Your Custom Category 2",
    # ... add more
]
```

## Data Structure

Transactions are stored with category information:

```json
{
  "booking_date": "30.12.25",
  "payer": "John Doe",
  "payee": "Amazon",
  "purpose": "Online purchase",
  "amount": -49.99,
  "category_info": {
    "category": "Shopping",
    "confidence": "high",
    "reasoning": "Online retail purchase from Amazon"
  }
}
```

## Performance

- **Model**: Gemma 2B (~1.6GB)
- **Speed**: ~2-5 seconds per batch of 5 transactions
- **Memory**: ~2GB RAM for model
- **Batch Size**: Adjustable (default: 5)

For 100 transactions:
- Expected time: 2-4 minutes
- Progress updates every batch

## Troubleshooting

### "Connection to server lost"
- Ensure Ollama is running: `ollama serve`
- Check if model is downloaded: `ollama list`

### "Model not found"
```bash
ollama pull gemma2:2b
```

### Slow performance
- Reduce batch size in `categorizer_service.py`
- Use a smaller model: `gemma:2b`
- Ensure sufficient RAM (minimum 4GB)

### Categories not appearing
- Refresh the page after categorization
- Check browser console for errors
- Verify transactions.json was updated

## Alternative Models

You can use different Ollama models:

```python
# Smaller, faster (1.5GB)
model = "gemma:2b"

# Larger, more accurate (7GB)
model = "gemma2:9b"

# Even more accurate (26GB)
model = "llama3.3:70b"
```

Change the model in:
- `CategorizationModal.tsx`: Update the `model=` parameter
- Or modify `categorizer_service.py`: Change the default model

## Testing

To test categorization:

1. Upload a CSV file
2. Click "Categorize" button
3. Monitor console logs for:
   ```
   [Backend] Processing batch 1-5 of 50...
   [Frontend] Progress: 10%
   ```
4. Verify categories appear in table after completion

## Files Modified/Created

### Created:
- `backend/services/categorizer_service.py`
- `backend/services/categorizer_router.py`
- `backend/requirements_categorizer.txt`
- `frontend/src/components/CategorizationModal.tsx`
- `CATEGORIZATION_SETUP.md` (this file)

### Modified:
- `backend/app.py` - Added categorizer router
- `frontend/src/pages/PersonalFinance.tsx` - Added categorize button
- `frontend/src/components/TransactionTable.tsx` - Added category column

## Future Enhancements

Potential improvements:
- [ ] Manual category editing
- [ ] Category suggestions based on history
- [ ] Custom category rules
- [ ] Category statistics and charts
- [ ] Export categorized data
- [ ] Batch categorize all uploads
- [ ] Category confidence filtering
