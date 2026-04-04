"""
Wrapper script to run the FastAPI app from the root directory
"""
import sys
import os

# Add customer-support-env to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'customer-support-env'))

# Now import - the try/except in app.py will handle whether to use relative or absolute imports
from app import app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=7860)
