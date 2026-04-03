#!/usr/bin/env python3
"""Script to fix all three files."""
import os

# Fix 1: models.py - add value and status fields
models_content = '''"""
Pydantic models for the SupportSentinelEnv environment.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    """Represents a single customer support ticket."""
    ticket_id: str
    subject: str
    body: str
    customer_name: str
    customer_tier: str = Field(..., pattern="^(free|pro|enterprise)$")
    category: str = Field(..., pattern="^(billing|technical|account|shipping|general)$")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sla_steps_remaining: int
    sla_total_steps: int
    interaction_count: int = 0
    sentiment_history: List[float] = []
    resolved: bool = False
    escalated: bool = False
    sla_breached: bool = False
    value: float = 1.0
    status: str = "pending"

    def model_post_init(self, __context: Any) -> None:
        """Ensure sentiment history starts with the initial score."""
        if not self.sentiment_history:
            self.sentiment_history.append(self.sentiment_score)


class Action(BaseModel):
    """Represents an action taken by the agent."""
    action_type: str = Field(..., pattern="^(prioritize|respond|escalate|compensate|resolve|defer)$")
    parameters: Dict[str, Any]


class Observation(BaseModel):
    """Represents the observation returned to the agent."""
    tickets: List[Ticket]
    task_id: str
    task_description: str
    step_number: int
    max_steps: int
    available_actions: List[str]
    current_score: float


class Reward(BaseModel):
    """Represents the reward for a step."""
    score: float
    partial_scores: Dict[str, float]
    feedback: str
    cumulative_score: float


class EnvState(BaseModel):
    """Represents the full state of the environment for a session."""
    session_id: str
    task_id: str
    step_number: int
    max_steps: int
    done: bool
    cumulative_score: float
    tickets: List[Ticket]


class StepResponse(BaseModel):
    """The combined response for a `step` action."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""
    task_id: str
    session_id: Optional[str] = None
    seed: int = 42
'''

# Fix 2: inference.py - complete rewrite
inference_content = '''"""
Baseline inference script for SupportSentinelEnv.
"""
import json
import uuid
import httpx
from datetime import datetime
from typing import List

API_BASE_URL = "http://localhost:7860"
http_client = httpx.Client(base_url=API_BASE_URL)

def log_event(event_type: str, **kwargs):
    """Logs a JSON event."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        **kwargs
    }
    print(json.dumps(log_entry), flush=True)

def get_action(task_id: str, observation: dict) -> dict:
    """Get action for a task based on observation."""
    if task_id == "sla_triage":
        ticket_ids = [t["ticket_id"] for t in observation["tickets"]]
        return {"action_type": "prioritize", "parameters": {"ticket_ids": ticket_ids}}
    
    elif task_id == "sentiment_recovery":
        ticket = observation["tickets"][0]
        return {"action_type": "respond", "parameters": {"ticket_id": ticket["ticket_id"], "tone": "empathetic"}}
    
    elif task_id == "queue_optimization":
        tickets = [t for t in observation["tickets"] if not t["resolved"]]
        if tickets:
            return {"action_type": "resolve", "parameters": {"ticket_id": tickets[0]["ticket_id"]}}
        return {"action_type": "defer", "parameters": {}}
    
    return {"action_type": "defer", "parameters": {}}

def run_episode(task_id: str, seed: int):
    """Run a single episode."""
    session_id = f"ep_{task_id}_{seed}_{uuid.uuid4().hex[:8]}"
    log_event("START", task=task_id, env="SupportSentinelEnv", model="baseline_agent")
    
    rewards = []
    step_count = 0
    
    try:
        # Reset
        response = http_client.post("/reset", json={"task_id": task_id, "session_id": session_id, "seed": seed})
        response.raise_for_status()
        observation = response.json()

        done = False
        while not done and step_count < 20:
            action = get_action(task_id, observation)
            
            # Step
            response = http_client.post(f"/step?session_id={session_id}", json=action)
            response.raise_for_status()
            
            result = response.json()
            observation = result["observation"]
            done = result["done"]
            
            reward_score = result["reward"]["score"]
            rewards.append(reward_score)
            step_count += 1

        score = sum(rewards)
        log_event("END", success=True, steps=step_count, score=f"{score:.3f}", rewards=rewards)

    except Exception as e:
        log_event("ERROR", message=str(e))
        log_event("END", success=False, steps=step_count, score="0.000", rewards=rewards)

def main():
    tasks = ["sla_triage", "sentiment_recovery", "queue_optimization"]
    seeds = [42, 43]

    for task_id in tasks:
        print(f"\\n--- Running Task: {task_id} ---")
        for seed in seeds:
            run_episode(task_id, seed)

if __name__ == "__main__":
    main()
'''

# Fix 3: Read and fix graders.py
graders_path = r"c:\Users\asus4\Desktop\Hackethon\customer-support-env\graders.py"
with open(graders_path, 'r') as f:
    graders_content = f.read()

# Replace the grade_queue_optimization function
old_func = '''def grade_queue_optimization(action: Dict[str, Any], initial_tickets: List[Ticket], final_tickets: List[Ticket], cumulative_score: float, done: bool) -> Reward:
    """
    Grades the 'queue_optimization' task.
    Score is based on the value of tickets resolved.
    A bonus is given for resolving tickets with higher value first.
    """
    # This grader is called at the end of the episode.
    # The action that leads to the 'done' state is the final 'resolve'.
    if not done:
        # Intermediate steps have no reward, but we must return a valid Reward object.
        return Reward(
            score=0.0,
            partial_scores={},
            feedback="Ticket resolved. Awaiting end of episode for final score.",
            cumulative_score=cumulative_score
        )

    # --- Final score calculation ---
    resolved_tickets = [t for t in final_tickets if t.status == "resolved"]
    
    if not resolved_tickets:
        return Reward(
            score=0.0,
            partial_scores={"total_value": 0},
            feedback="No tickets were resolved.",
            cumulative_score=cumulative_score
        )

    total_value = sum(t.value for t in resolved_tickets)
    
    # Normalize the score against the total possible value from the initial set of tickets
    score = total_value / max_possible_value if max_possible_value > 0 else 0

    return Reward(
        score=score,
        partial_scores={"total_value": total_value},
        feedback=f"Episode finished. Total value of resolved tickets: {total_value}",
        cumulative_score=cumulative_score + score
    )'''

new_func = '''def grade_queue_optimization(action: Dict[str, Any], initial_tickets: List[Ticket], final_tickets: List[Ticket], cumulative_score: float, done: bool, max_steps: int) -> Reward:
    """
    Grades the 'queue_optimization' task.
    Score is based on the value of tickets resolved.
    """
    if not done:
        return Reward(
            score=0.0,
            partial_scores={},
            feedback="Ticket processed.",
            cumulative_score=cumulative_score
        )

    resolved_tickets = [t for t in final_tickets if t.resolved]
    
    if not resolved_tickets:
        return Reward(
            score=0.0,
            partial_scores={"total_value": 0},
            feedback="No tickets resolved.",
            cumulative_score=cumulative_score
        )

    total_value = sum(t.value for t in resolved_tickets)
    max_possible_value = sum(t.value for t in initial_tickets)
    
    score = total_value / max_possible_value if max_possible_value > 0 else 0.0
    score = min(score, 1.0)

    return Reward(
        score=score,
        partial_scores={"total_value": total_value, "max_value": max_possible_value},
        feedback=f"Resolved {len(resolved_tickets)} tickets worth {total_value:.1f}/{max_possible_value:.1f}",
        cumulative_score=cumulative_score + score
    )'''

updated_graders = graders_content.replace(old_func, new_func)

# Write all three files
with open(r"c:\Users\asus4\Desktop\Hackethon\customer-support-env\models.py", 'w') as f:
    f.write(models_content)
print("✓ Fixed models.py")

with open(r"c:\Users\asus4\Desktop\Hackethon\customer-support-env\inference.py", 'w') as f:
    f.write(inference_content)
print("✓ Fixed inference.py")

with open(graders_path, 'w') as f:
    f.write(updated_graders)
print("✓ Fixed graders.py")

print("\nAll files have been corrected!")
