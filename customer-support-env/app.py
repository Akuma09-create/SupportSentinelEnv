"""
FastAPI server for the SupportSentinelEnv OpenEnv environment.
"""
import uuid
from collections import OrderedDict
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from typing import Optional

from .models import Action, Observation, StepResponse, EnvState, ResetRequest
from .environment import SupportSentinelEnv
from .tasks import TASK_DEFINITIONS

# --- App Initialization ---
app = FastAPI(
    title="SupportSentinelEnv",
    description="An OpenEnv-compliant environment for AI Customer Support.",
    version="1.0.0",
)

# --- In-Memory Session Management ---
MAX_SESSIONS = 500
# OrderedDict can be used as a simple LRU cache
sessions: OrderedDict[str, SupportSentinelEnv] = OrderedDict()

def get_session(session_id: str) -> SupportSentinelEnv:
    """
    Retrieves an environment instance from the session cache.
    Raises HTTPException if the session is not found.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    # Move session to the end to mark it as recently used
    sessions.move_to_end(session_id)
    return sessions[session_id]

def create_session(task_id: str, seed: int, session_id: str = None) -> SupportSentinelEnv:
    """
    Creates a new environment instance and adds it to the session cache.
    Manages the LRU cache size.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    if session_id in sessions:
        raise HTTPException(status_code=409, detail=f"Session '{session_id}' already exists.")

    if len(sessions) >= MAX_SESSIONS:
        # Evict the least recently used session
        oldest_session_id, _ = sessions.popitem(last=False)
        print(f"Max sessions reached. Evicting oldest session: {oldest_session_id}")

    env = SupportSentinelEnv(task_id=task_id, seed=seed)
    sessions[session_id] = env
    return env

# --- API Endpoints ---

@app.get("/health", tags=["General"])
async def health_check():
    """Returns a 200 OK status to indicate the server is running."""
    return {"status": "ok"}

@app.get("/tasks", tags=["Tasks"])
async def list_tasks() -> Dict[str, Dict]:
    """Returns a list of all available tasks with their descriptions."""
    return {
        task_id: {
            "description": details["description"],
            "difficulty": details["difficulty"],
            "max_steps": details["max_steps"],
            "available_actions": details["available_actions"],
        }
        for task_id, details in TASK_DEFINITIONS.items()
    }

@app.post("/reset", response_model=Observation, tags=["Environment"])
async def reset_environment(
    request_body: Optional[ResetRequest] = Body(None),
    task_id: str = Query("sla_triage"),
    session_id: str = Query(None),
    seed: int = Query(42)
):
    """
    Resets or creates an environment. Accepts POST with an optional body.
    If body is present, it's used. Otherwise, query parameters are used.
    This handles the validator sending a POST with an empty body.
    """
    try:
        # Determine which values to use
        final_task_id = task_id
        final_session_id = session_id
        final_seed = seed

        if request_body:
            final_task_id = request_body.task_id or task_id
            final_session_id = request_body.session_id or session_id
            final_seed = request_body.seed or seed

        env = create_session(final_task_id, final_seed, final_session_id)
        observation = env.reset()

        # Find assigned session_id to return in headers
        assigned_id = None
        for sid, e in sessions.items():
            if e is env:
                assigned_id = sid
                break
        
        headers = {"X-Session-Id": assigned_id} if assigned_id else {}
        return JSONResponse(content=observation.dict(), headers=headers)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def take_step(
    action: Action,
    session_id: str = Query(..., description="The unique identifier for the environment session.")
) -> StepResponse:
    """
    Takes a step in the environment using the provided action.
    Returns the new observation, reward, done flag, and info.
    """
    try:
        env = get_session(session_id)
        observation, reward, done, info = env.step(action)
        
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during step: {e}")


@app.get("/state", response_model=EnvState, tags=["Environment"])
async def get_environment_state(
    session_id: str = Query(..., description="The unique identifier for the environment session.")
) -> EnvState:
    """
    Retrieves the complete current state of a specific environment session.
    """
    try:
        env = get_session(session_id)
        return env.get_state(session_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
