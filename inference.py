"""
Baseline inference script for SupportSentinelEnv.
OpenEnv-compliant with LiteLLM proxy integration.
"""
import json
import os
import sys
import uuid
import httpx
import time
from functools import wraps
from typing import List, Optional, Any

from openai import OpenAI

# Configuration - API_BASE_URL and API_KEY MUST be provided by validator
API_BASE_URL = os.environ.get("API_BASE_URL")
if API_BASE_URL is None:
    raise ValueError("API_BASE_URL environment variable is required (LiteLLM proxy endpoint)")

API_KEY = os.environ.get("API_KEY")
if API_KEY is None:
    raise ValueError("API_KEY environment variable is required")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# OpenAI client (configured to use LiteLLM proxy)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# HTTP client
http_client = httpx.Client(base_url=ENV_BASE_URL, timeout=30.0)

# System prompts for each task
SYSTEM_PROMPTS = {
    "sla_triage": "You are a customer support ticket triage specialist. Analyze tickets and prioritize them based on SLA urgency. Return a JSON action object.",
    "sentiment_recovery": "You are a customer recovery specialist. Determine the best sequence of compensations and responses to recover customer sentiment. Return a JSON action object.",
    "queue_optimization": "You are a queue optimization specialist. Resolve tickets in the optimal order considering SLA and value. Return a JSON action object.",
}


def format_observation(task_id: str, observation: dict) -> str:
    """Format observation into a readable prompt for the LLM."""
    task_info = f"Task: {task_id}\n"
    obs_info = f"Current observation: {json.dumps(observation, indent=2)}\n"
    
    if task_id == "sla_triage":
        return f"{task_info}Analyze the tickets and return a JSON action to prioritize them by SLA urgency.\n{obs_info}Return only valid JSON."
    elif task_id == "sentiment_recovery":
        return f"{task_info}Determine the best compensation and response steps to recover customer sentiment.\n{obs_info}Return only valid JSON."
    elif task_id == "queue_optimization":
        return f"{task_info}Resolve tickets in optimal order considering SLA and value.\n{obs_info}Return only valid JSON."
    else:
        return f"{task_info}{obs_info}Return only valid JSON."


def fmt_bool(value: bool) -> str:
    """Format boolean as lowercase string."""
    return str(value).lower()


def fmt_reward(value: float) -> str:
    """Format reward to 2 decimal places."""
    return f"{float(value):.2f}"


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step_num: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log individual step."""
    error_str = error if error is not None else "null"
    print(
        f"[STEP] step={step_num} action={action} reward={fmt_reward(reward)} "
        f"done={fmt_bool(done)} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """Log episode end."""
    rewards_str = ",".join(fmt_reward(r) for r in rewards)
    print(
        f"[END] success={fmt_bool(success)} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator for network calls."""
    def decorator(func):
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

def get_fallback_action(task_id: str, observation: dict) -> dict:
    """Fallback rule-based action when LLM fails."""
    if task_id == "sla_triage":
        sorted_ids = sorted(
            [t["ticket_id"] for t in observation.get("tickets", [])], 
            key=lambda tid: next((t["sla_steps_remaining"] for t in observation.get("tickets", []) if t["ticket_id"] == tid), 999)
        )
        return {"action_type": "prioritize", "parameters": {"ticket_ids": sorted_ids}}
    elif task_id == "sentiment_recovery":
        tickets = observation.get("tickets", [])
        if tickets:
            return {"action_type": "compensate", "parameters": {"ticket_id": tickets[0]["ticket_id"], "type": "refund"}}
    elif task_id == "queue_optimization":
        unresolved = [t for t in observation.get("tickets", []) if not t.get("resolved", False)]
        if unresolved:
            return {"action_type": "resolve", "parameters": {"ticket_id": unresolved[0]["ticket_id"]}}
    return {"action_type": "defer", "parameters": {}}


def get_action(task_id: str, observation: dict, step: int = 0) -> dict:
    """Get next action using LLM with fallback to rule-based logic."""
    try:
        # Format observation for LLM
        user_prompt = format_observation(task_id, observation)
        system_prompt = SYSTEM_PROMPTS.get(task_id, "You are a helpful AI agent. Return only valid JSON.")
        
        # Call LLM (will be routed through LiteLLM proxy via API_BASE_URL)
        print(f"[DEBUG] Calling LLM for task {task_id} with base_url={API_BASE_URL[:50]}...", file=sys.stderr, flush=True)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        print(f"[DEBUG] LLM response received for task {task_id}", file=sys.stderr, flush=True)
        
        # Parse LLM response
        action_text = response.choices[0].message.content.strip()
        try:
            action = json.loads(action_text)
            return action
        except json.JSONDecodeError:
            # If JSON parsing fails, use fallback
            return get_fallback_action(task_id, observation)
            
    except Exception as e:
        # If LLM call fails, log it and use fallback
        print(f"[ERROR] LLM call failed: {type(e).__name__}: {str(e)}", file=sys.stderr, flush=True)
        return get_fallback_action(task_id, observation)

def run_episode(task_id: str, seed: int) -> None:
    """Run a single episode."""
    session_id = f"ep_{task_id}_{seed}_{uuid.uuid4().hex[:8]}"
    log_start(task_id, "SupportSentinelEnv", MODEL_NAME)
    
    rewards: List[float] = []
    step_count = 0
    success = False
    last_error = None
    
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
                last_error = reward_obj.get("last_action_error")
            elif isinstance(reward_obj, (int, float)):
                reward_score = float(reward_obj)
                last_error = None
            else:
                reward_score = 0.0
                last_error = None
            
            rewards.append(reward_score)
            log_step(step_count + 1, action_json, reward_score, done, last_error)
            step_count += 1
        
        # Success only if episode completed (done=True)
        success = done and step_count > 0
    
    except Exception as e:
        success = False
    
    finally:
        log_end(success, step_count, rewards)


def main():
    """Run all episodes."""
    for task_id in ["sla_triage", "sentiment_recovery", "queue_optimization"]:
        for seed in [42, 43]:
            run_episode(task_id, seed)


if __name__ == "__main__":
    main()
