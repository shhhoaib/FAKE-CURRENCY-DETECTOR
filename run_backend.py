#!/usr/bin/env python3
"""Run FastAPI backend in correct directory."""
import os
import sys
import uvicorn

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Run uvicorn without reload for Windows compatibility
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
