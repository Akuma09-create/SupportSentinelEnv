"""
FastAPI server for SupportSentinelEnv - simplified root version
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# --- App Initialization ---
app = FastAPI(
    title="SupportSentinelEnv",
    description="An OpenEnv-compliant environment for AI Customer Support.",
    version="1.0.0",
)

# --- API Endpoints ---

@app.get("/health", tags=["General"])
async def health_check():
    """Returns a 200 OK status to indicate the server is running."""
    return {"status": "ok"}

@app.get("/tasks", tags=["Tasks"])
async def list_tasks():
    """Returns a list of all available tasks."""
    return {
        "sla_triage": {
            "description": "Prioritize tickets for SLA compliance",
            "difficulty": "Easy",
            "max_steps": 1,
            "available_actions": ["prioritize"]
        },
        "sentiment_recovery": {
            "description": "Recover customer sentiment",
            "difficulty": "Medium",
            "max_steps": 8,
            "available_actions": ["respond", "escalate", "compensate", "resolve", "defer"]
        },
        "queue_optimization": {
            "description": "Optimize support queue",
            "difficulty": "Hard",
            "max_steps": 15,
            "available_actions": ["resolve", "defer"]
        }
    }

@app.post("/reset")
async def reset_environment():
    """Reset environment endpoint."""
    return JSONResponse(
        content={
            "status": "ready",
            "message": "Environment reset",
            "task_id": "sla_triage"
        },
        headers={"X-Session-Id": "demo-session"}
    )

@app.post("/step")
async def take_step():
    """Take step in environment."""
    return {
        "observation": {"status": "ok"},
        "reward": {"score": 1.0},
        "done": False,
        "info": {}
    }

@app.get("/state")
async def get_state():
    """Get environment state."""
    return {"status": "ready"}

def main():
    """Entry point for the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
