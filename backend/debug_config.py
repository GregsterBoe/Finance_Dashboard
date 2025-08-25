# Debug configuration for FastAPI backend
# This file shows you how to set up debugging for your FastAPI application

import debugpy

def configure_debug_mode():
    """
    Call this function to enable remote debugging.
    Useful when running the app with uvicorn --reload
    """
    # Enable debugpy server
    debugpy.listen(("localhost", 5678))
    print("⏳ Waiting for debugger to attach...")
    debugpy.wait_for_client()  # blocks execution until client is attached
    print("🐛 Debugger attached!")

# Example usage in your app.py:
# from debug_config import configure_debug_mode
# configure_debug_mode()  # Add this line at the start of your app
