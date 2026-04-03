"""
An automated walkthrough script for Task 2: Sentiment Recovery.

This script demonstrates a successful path through the task by programmatically
interacting with the SupportSentinelEnv API.
"""
import os
import json
import time
import requests

# --- Configuration ---
API_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
TASK_ID = "sentiment_recovery"
SEED = 42

def print_step_info(step_name: str, action: dict, reward: dict):
    """Prints formatted information about a step."""
    print(f"\n--- {step_name} ---")
    print(f"Action: {json.dumps(action)}")
    print(f"Feedback: {reward['feedback']}")
    print(f"Score Received: {reward['score']:.2f}")
    print(f"New Cumulative Score: {reward['cumulative_score']:.2f}")
    print("-" * (len(step_name) + 8))

def main():
    """Main function to run the automated walkthrough."""
    print("--- Automated Walkthrough for Task 2: Sentiment Recovery ---")

    # 1. Check server health
    try:
        requests.get(f"{API_BASE_URL}/health").raise_for_status()
        print("Environment server is running.")
    except requests.RequestException:
        print(f"\n[ERROR] Could not connect to the server at {API_BASE_URL}.")
        print("Please ensure the server is running in another terminal:")
        print("uvicorn customer-support-env.app:app --port 7860")
        return

    # 2. Reset the environment
    try:
        response = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_id": TASK_ID, "seed": SEED}
        )
        response.raise_for_status()
        observation = response.json()
        session_id = response.headers['X-Session-Id']
        print(f"Started new session. Session ID: {session_id}")
        initial_sentiment = observation['tickets'][0]['sentiment_score']
        print(f"Initial sentiment: {initial_sentiment:.2f}")
    except requests.RequestException as e:
        print(f"[ERROR] Failed to reset environment: {e}")
        return

    time.sleep(1) # Pause for readability

    # --- Define the sequence of actions ---
    actions = [
        {
            "name": "Step 1: Apologize",
            "action": {"action_type": "respond", "parameters": {"ticket_id": "t_angry", "tone": "apologetic"}}
        },
        {
            "name": "Step 2: Compensate with Refund",
            "action": {"action_type": "compensate", "parameters": {"ticket_id": "t_angry", "type": "refund"}}
        },
        {
            "name": "Step 3: Show Empathy",
            "action": {"action_type": "respond", "parameters": {"ticket_id": "t_angry", "tone": "empathetic"}}
        },
        {
            "name": "Step 4: Resolve the Ticket",
            "action": {"action_type": "resolve", "parameters": {"ticket_id": "t_angry", "note": "Refund processed. Issue resolved."}}
        }
    ]

    # 3. Execute the actions
    done = False
    for step_data in actions:
        if done:
            print("Episode finished early.")
            break

        action = step_data['action']
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/step?session_id={session_id}",
                json=action
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            print(f"\n[ERROR] Failed to execute step: {e.response.json()['detail']}")
            return

        print_step_info(step_data['name'], action, result['reward'])
        done = result['done']
        final_observation = result['observation']
        time.sleep(1.5) # Pause for readability

    # 4. Print final results
    print("\n--- Episode Finished ---")
    final_ticket = final_observation['tickets'][0]
    final_score = result['reward']['cumulative_score']
    
    print(f"Final Score: {final_score:.2f}")
    print(f"Total Steps: {final_observation['step_number']}")
    print(f"Final Sentiment: {final_ticket['sentiment_score']:.2f} (Goal was > 0.3)")
    print(f"SLA Steps Remaining: {final_ticket['sla_steps_remaining']}")
    print(f"Ticket Resolved: {'Yes' if final_ticket['resolved'] else 'No'}")
    print(f"SLA Breached: {'Yes' if final_ticket['sla_breached'] else 'No'}")


if __name__ == "__main__":
    main()
