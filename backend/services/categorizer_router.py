"""
API router for transaction categorization.
Provides endpoints to categorize transactions with real-time progress updates.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import json
import asyncio
from typing import AsyncGenerator
import queue
import threading

from services.categorizer_service import get_categorizer_service

router = APIRouter()

# Base directory for uploads
BASE_UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"


class CategorizationRequest(BaseModel):
    """Request to categorize transactions for an upload."""
    upload_id: str
    model: str = "gemma3:4b"


class CategorizationResponse(BaseModel):
    """Response after successful categorization."""
    upload_id: str
    total_categorized: int
    category_counts: dict
    categories: list


class ProgressUpdate(BaseModel):
    """Progress update during categorization."""
    current: int
    total: int
    status: str
    percentage: float


async def categorize_with_progress(
    upload_id: str,
    model: str = "gemma3:4b"
) -> AsyncGenerator[str, None]:
    """
    Generator that yields Server-Sent Events for categorization progress.

    Args:
        upload_id: UUID of the upload to categorize
        model: Ollama model to use

    Yields:
        SSE-formatted progress updates
    """
    upload_dir = BASE_UPLOAD_DIR / upload_id

    if not upload_dir.exists():
        yield f"data: {json.dumps({'error': 'Upload not found'})}\n\n"
        return

    # Use thread-safe queue for progress updates
    progress_queue = queue.Queue()

    def progress_callback(current: int, total: int, status: str):
        """Callback called by categorizer service with progress updates."""
        percentage = (current / total * 100) if total > 0 else 0
        progress_data = {
            "type": "progress",
            "current": current,
            "total": total,
            "status": status,
            "percentage": round(percentage, 1)
        }
        # Put in thread-safe queue
        progress_queue.put(progress_data)

    # Start categorization in background
    categorizer = get_categorizer_service(model=model)

    async def run_categorization():
        """Run categorization synchronously in executor."""
        try:
            result = await asyncio.to_thread(
                categorizer.categorize_upload,
                upload_dir,
                False,
                progress_callback
            )
            progress_queue.put({"type": "complete", **result})
        except Exception as e:
            progress_queue.put({
                "type": "error",
                "error": str(e)
            })

    # Start categorization task
    task = asyncio.create_task(run_categorization())

    # Yield progress updates as they come in
    while True:
        try:
            # Check for updates without blocking
            try:
                update = progress_queue.get_nowait()
                # Send update as SSE
                yield f"data: {json.dumps(update)}\n\n"

                # If complete or error, break
                if update.get("type") in ["complete", "error"]:
                    break
            except queue.Empty:
                # No update available, send keepalive
                yield f": keepalive\n\n"
                await asyncio.sleep(0.5)

            # Check if task is done
            if task.done():
                # Check for any remaining messages
                while not progress_queue.empty():
                    update = progress_queue.get_nowait()
                    yield f"data: {json.dumps(update)}\n\n"
                break

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            break

    # Ensure task is complete
    if not task.done():
        await task


@router.get("/categorize/stream/{upload_id}")
async def categorize_upload_stream(upload_id: str, model: str = "gemma3:4b"):
    """
    Categorize transactions for an upload with real-time progress updates via SSE.

    - **upload_id**: UUID of the upload to categorize
    - **model**: Ollama model to use (default: gemma3:4b)

    Returns Server-Sent Events with progress updates.
    """
    return StreamingResponse(
        categorize_with_progress(upload_id, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/categorize/{upload_id}", response_model=CategorizationResponse)
async def categorize_upload(upload_id: str, model: str = "gemma3:4b"):
    """
    Categorize transactions for an upload (non-streaming version).

    - **upload_id**: UUID of the upload to categorize
    - **model**: Ollama model to use (default: gemma3:4b)

    Returns categorization results.
    """
    upload_dir = BASE_UPLOAD_DIR / upload_id

    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")

    try:
        categorizer = get_categorizer_service(model=model)
        result = await asyncio.to_thread(
            categorizer.categorize_upload,
            upload_dir
        )

        return CategorizationResponse(
            upload_id=upload_id,
            **result
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error categorizing transactions: {str(e)}"
        )


@router.get("/categories")
async def get_available_categories():
    """
    Get list of available transaction categories.

    Returns list of category names.
    """
    categorizer = get_categorizer_service()
    return {"categories": categorizer.categories}
