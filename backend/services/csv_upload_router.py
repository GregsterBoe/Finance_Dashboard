from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from services.csv_handler import (
    get_csv_handler_service,
    UploadResponse,
    UploadListResponse,
    UploadMetadata,
    TransactionListResponse
)
import tempfile
import os

router = APIRouter()


@router.post("/csv/upload", response_model=UploadResponse)
async def upload_csv_file(file: UploadFile = File(...)):
    """
    Upload and parse a CSV file (German bank export format)

    - **file**: CSV file to upload

    Returns upload metadata with summary statistics
    """
    print(f"[DEBUG] Received file: {file.filename}")
    print(f"[DEBUG] Content type: {file.content_type}")

    # Validate file extension
    if not file.filename or not file.filename.endswith('.csv'):
        print(f"[DEBUG] File validation failed: filename={file.filename}")
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        # Read file content
        content = await file.read()
        print(f"[DEBUG] File size: {len(content)} bytes")

        # Try different encodings
        try:
            file_content = content.decode('utf-8')
            print("[DEBUG] Successfully decoded with UTF-8")
        except UnicodeDecodeError:
            try:
                file_content = content.decode('latin-1')
                print("[DEBUG] Successfully decoded with Latin-1")
            except UnicodeDecodeError:
                file_content = content.decode('cp1252')
                print("[DEBUG] Successfully decoded with CP1252")

        print(f"[DEBUG] File content preview (first 200 chars): {file_content[:200]}")

        # Process upload
        csv_service = get_csv_handler_service()
        result = csv_service.process_upload(file_content, file.filename)

        print(f"[DEBUG] Upload successful: {result.upload_id}")
        return result

    except ValueError as e:
        print(f"[DEBUG] ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except UnicodeDecodeError as e:
        print(f"[DEBUG] UnicodeDecodeError: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Unable to decode file. Please ensure it's a valid CSV file with UTF-8, Latin-1, or Windows-1252 encoding. Error: {str(e)}")
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/csv/uploads", response_model=UploadListResponse)
async def get_all_uploads():
    """
    Get list of all uploaded files with summary statistics

    Returns list of uploads sorted by most recent first
    """
    try:
        csv_service = get_csv_handler_service()
        return csv_service.get_all_uploads()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving uploads: {str(e)}")


@router.get("/csv/uploads/{upload_id}", response_model=UploadMetadata)
async def get_upload_details(upload_id: str):
    """
    Get metadata and summary for a specific upload

    - **upload_id**: Unique identifier for the upload

    Returns upload metadata with summary statistics
    """
    try:
        csv_service = get_csv_handler_service()
        return csv_service.get_upload_metadata(upload_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving upload details: {str(e)}")


@router.get("/csv/uploads/{upload_id}/transactions", response_model=TransactionListResponse)
async def get_upload_transactions(
    upload_id: str,
    offset: int = Query(0, ge=0, description="Number of transactions to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of transactions to return")
):
    """
    Get transactions for a specific upload with pagination

    - **upload_id**: Unique identifier for the upload
    - **offset**: Number of transactions to skip (for pagination)
    - **limit**: Maximum number of transactions to return (1-1000)

    Returns paginated list of transactions
    """
    try:
        csv_service = get_csv_handler_service()
        return csv_service.get_transactions(upload_id, offset, limit)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving transactions: {str(e)}")


@router.delete("/csv/uploads/{upload_id}")
async def delete_upload(upload_id: str):
    """
    Delete an upload and all its associated data

    - **upload_id**: Unique identifier for the upload

    Returns success message
    """
    try:
        csv_service = get_csv_handler_service()
        csv_service.delete_upload(upload_id)
        return JSONResponse(
            status_code=200,
            content={"message": f"Upload {upload_id} deleted successfully"}
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting upload: {str(e)}")
