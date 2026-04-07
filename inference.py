"""
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

def get_action(task_id: str, observation: dict, step: int = 0) -> dict:
    """Get action for a task based on observation with intelligent strategy."""
    if task_id == "sla_triage":
        # Sort tickets by SLA urgency (ascending SLA time = highest priority)
        sorted_ids = sorted(
            [t["ticket_id"] for t in observation["tickets"]], 
            key=lambda tid: next((t["sla_steps_remaining"] for t in observation["tickets"] if t["ticket_id"] == tid), 999)
        )
        return {"action_type": "prioritize", "parameters": {"ticket_ids": sorted_ids}}
    
    elif task_id == "sentiment_recovery":
        # Intelligent strategy based on sentiment level
        ticket = observation["tickets"][0]
        sentiment = ticket["sentiment_score"]
        steps_remaining = observation.get("max_steps", 8) - step
        
        if sentiment < -0.6:  # Very angry - escalate
            return {"action_type": "escalate", "parameters": {"ticket_id": ticket["ticket_id"]}}
        elif sentiment < -0.3:  # Angry - compensate 
            return {"action_type": "compensate", "parameters": {"ticket_id": ticket["ticket_id"], "type": "refund"}}
        elif sentiment < 0.2:  # Unhappy - respond
            return {"action_type": "respond", "parameters": {"ticket_id": ticket["ticket_id"], "tone": "solution_focused"}}
        elif steps_remaining <= 2 and sentiment < 0.3:  # Last chance
            return {"action_type": "compensate", "parameters": {"ticket_id": ticket["ticket_id"], "type": "priority_support"}}
        else:  # Good sentiment or final step
            return {"action_type": "resolve", "parameters": {"ticket_id": ticket["ticket_id"]}}
    
    elif task_id == "queue_optimization":
        # Resolve high-value tickets first
        tickets = [t for t in observation["tickets"] if not t["resolved"]]
        if tickets:
            # Sort by value (highest first)
            tickets.sort(key=lambda t: t.get("value", 0), reverse=True)
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
            action = get_action(task_id, observation, step_count)
            
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
        print(f"\n--- Running Task: {task_id} ---")
        for seed in seeds:
            run_episode(task_id, seed)

if __name__ == "__main__":
    main()
