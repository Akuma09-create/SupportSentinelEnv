"""
Baseline inference script for SupportSentinelEnv using the OpenAI client.
This script is updated to be compliant with the latest competition logging format.
"""
import os
import json
import uuid
import requests
from typing import List, Optional
from openai import OpenAI

# --- Configuration ---
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
HF_TOKEN = os.environ.get("HF_TOKEN", "your-hf-token-if-needed")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
BENCHMARK_NAME = "SupportSentinelEnv"
SUCCESS_SCORE_THRESHOLD = 0.6 # Define a success threshold for the [END] log

# --- System Prompt ---
SYSTEM_PROMPT = """You are an expert customer support manager AI.
You manage a queue of customer tickets. Each ticket has a sentiment score (-1.0=furious, +1.0=happy) and an SLA deadline in steps.
Your goal: resolve all tickets with positive sentiment before SLA expires.
Always respond with ONLY a JSON object:
{"action_type": "...", "parameters": {...}}
Never add explanation or markdown."""

# --- OpenAI Client Initialization ---
client = OpenAI(
    base_url=f"{API_BASE_URL}/v1",
    api_key=HF_TOKEN,
)

# --- Logging Functions (New Format) ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action='{action}' reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_user_prompt(observation: dict) -> str:
    """Formats the observation into a user-friendly prompt for the LLM."""
    prompt = "Current Environment State:\n"
    prompt += f"Task: {observation['task_id']} ({observation['task_description']})\n"
    prompt += f"Step: {observation['step_number']}/{observation['max_steps']}\n"
    prompt += f"Available Actions: {', '.join(observation['available_actions'])}\n\n"
    prompt += "Tickets:\n"
    for t in observation['tickets']:
        if not t['resolved']:
            prompt += (
                f"- ID: {t['ticket_id']}, Tier: {t['customer_tier']}, "
                f"Sentiment: {t['sentiment_score']:.2f}, SLA: {t['sla_steps_remaining']} steps left, "
                f"Status: {'SLA Breached' if t['sla_breached'] else 'Active'}\n"
                f"  Subject: {t['subject']}\n"
            )
    prompt += "\nBased on the current state, what is your next action? Provide ONLY the JSON for your action."
    return prompt

def run_episode(task_id: str, seed: int):
    """Runs a single episode for a given task and seed, with new logging."""
    
    session_id = f"ep_{task_id}_{seed}_{uuid.uuid4().hex[:8]}"
    all_rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        # 1. Reset the environment
        response = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_id": task_id, "seed": seed, "session_id": session_id}
        )
        response.raise_for_status()
        observation = response.json()
        session_id = response.headers['X-Session-Id']

        done = False
        while not done:
            steps_taken += 1
            user_prompt = get_user_prompt(observation)
            
            # 2. Get action from LLM
            try:
                chat_completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                action_str = chat_completion.choices[0].message.content
                action = json.loads(action_str)
            except Exception as e:
                action = {"action_type": "defer", "parameters": {"ticket_id": observation['tickets'][0]['ticket_id']}}
                action_str = json.dumps(action)

            # 3. Take a step in the environment
            step_response = requests.post(
                f"{API_BASE_URL}/step?session_id={session_id}",
                json=action
            )
            step_response.raise_for_status()
            step_data = step_response.json()

            observation = step_data['observation']
            reward_info = step_data['reward']
            done = step_data['done']
            
            step_reward = reward_info.get('score', 0.0)
            all_rewards.append(step_reward)

            # Log the step in the new format
            log_step(step=steps_taken, action=json.dumps(action), reward=step_reward, done=done, error=None)

            if done:
                final_score = reward_info.get('cumulative_score', sum(all_rewards))
                break
        
        final_score = max(0.0, min(1.0, final_score)) # Clamp score
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] An error occurred during the episode: {e}")
        success = False
    finally:
        # Log the end of the episode
        log_end(success=success, steps=steps_taken, score=final_score, rewards=all_rewards)


def main():
    """Main function to run inference across all tasks."""
    tasks = ["sla_triage", "sentiment_recovery", "queue_optimization"]
    seeds = [42, 43]

    for task_id in tasks:
        print(f"\n--- Running Task: {task_id} ---")
        for seed in seeds:
            run_episode(task_id, seed)

if __name__ == "__main__":
    main()
