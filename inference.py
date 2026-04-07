"""
Baseline inference script for SupportSentinelEnv.
"""
import json
import os
import uuid
import httpx
import time
from functools import wraps
from typing import List, Optional

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# Validate HF_TOKEN
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is required")

# OpenAI client
from openai import OpenAI
client = OpenAI(base_url=API_BASE_URL, api_key=hf_token)

# HTTP client
http_client = httpx.Client(base_url=ENV_BASE_URL, timeout=30.0)

def log_start(task_id: str, env_name: str, model: str) -> None:
    print(f"[START] task={task_id} env={env_name} model={model}", flush=True)

def log_step(step_num: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_str = f"error={error}" if error else "error=null"
    print(f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} {error_str}", flush=True)

def log_end(success: bool, total_steps: int, rewards_list: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    print(f"[END] success={str(success).lower()} steps={total_steps} rewards={rewards_str}", flush=True)

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

def get_action(task_id: str, observation: dict, step: int = 0) -> dict:
    if task_id == "sla_triage":
        sorted_ids = sorted(
            [t["ticket_id"] for t in observation["tickets"]], 
            key=lambda tid: next((t["sla_steps_remaining"] for t in observation["tickets"] if t["ticket_id"] == tid), 999)
        )
        return {"action_type": "prioritize", "parameters": {"ticket_ids": sorted_ids}}
    
    elif task_id == "sentiment_recovery":
        ticket = observation["tickets"][0]
        if step == 0:
            return {"action_type": "compensate", "parameters": {"ticket_id": ticket["ticket_id"], "type": "refund"}}
        elif step == 1:
            return {"action_type": "compensate", "parameters": {"ticket_id": ticket["ticket_id"], "type": "free_month"}}
        elif step == 2:
            return {"action_type": "respond", "parameters": {"ticket_id": ticket["ticket_id"], "tone": "apologetic"}}
        else:
            return {"action_type": "resolve", "parameters": {"ticket_id": ticket["ticket_id"]}}
    
    elif task_id == "queue_optimization":
        unresolved = [t for t in observation["tickets"] if not t["resolved"]]
        if not unresolved:
            return {"action_type": "defer", "parameters": {}}
        prioritized = sorted(unresolved, key=lambda t: (t["sla_steps_remaining"], -t.get("value", 0)))
        return {"action_type": "resolve", "parameters": {"ticket_id": prioritized[0]["ticket_id"]}}
    
    return {"action_type": "defer", "parameters": {}}

def run_episode(task_id: str, seed: int):
    session_id = f"ep_{task_id}_{seed}_{uuid.uuid4().hex[:8]}"
    log_start(task_id, "SupportSentinelEnv", MODEL_NAME)
    
    rewards = []
    step_count = 0
    success = False
    
    try:
        @retry_on_failure(max_attempts=3, delay=1.0)
        def reset_with_retry():
            response = http_client.post("/reset", json={"task_id": task_id, "session_id": session_id, "seed": seed})
            response.raise_for_status()
            return response.json()
        
        observation = reset_with_retry()
        done = False
        
        while not done and step_count < 20:
            action = get_action(task_id, observation, step_count)
            action_json = json.dumps(action)
            
            @retry_on_failure(max_attempts=3, delay=1.0)
            def step_with_retry():
                response = http_client.post(f"/step?session_id={session_id}", json=action)
                response.raise_for_status()
                return response.json()
            
            result = step_with_retry()
            observation = result.get("observation", {})
            done = result.get("done", False)
            
            reward_obj = result.get("reward", {})
            if isinstance(reward_obj, dict):
                reward_score = float(reward_obj.get("score", 0))
            elif isinstance(reward_obj, (int, float)):
                reward_score = float(reward_obj)
            else:
                reward_score = 0.0
            
            rewards.append(reward_score)
            log_step(step_count + 1, action_json, reward_score, done)
            step_count += 1
        
        success = True
    
    except Exception:
        success = False
    
    finally:
        log_end(success=success, total_steps=step_count, rewards_list=rewards)

def main():
    for task_id in ["sla_triage", "sentiment_recovery", "queue_optimization"]:
        for seed in [42, 43]:
            run_episode(task_id, seed)

if __name__ == "__main__":
    main()
