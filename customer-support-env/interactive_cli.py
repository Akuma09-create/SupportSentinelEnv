"""
An interactive Command-Line Interface (CLI) to manually interact with the SupportSentinelEnv.
"""
import os
import json
import requests

# --- Configuration ---
API_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

def print_observation(obs: dict):
    """Prints the observation in a readable format."""
    print("\n" + "="*50)
    print(f" TASK: {obs['task_id']} | STEP: {obs['step_number']}/{obs['max_steps']} | SCORE: {obs['current_score']:.2f}")
    print("-"*50)
    print(f" Description: {obs['task_description']}")
    print(f" Available Actions: {', '.join(obs['available_actions'])}")
    print("-"*50)
    print(" TICKETS IN QUEUE:")
    
    if not obs['tickets']:
        print("  No tickets in the queue.")
        return

    for t in obs['tickets']:
        status_parts = []
        if t['resolved']:
            status_parts.append("✅ Resolved")
        if t['sla_breached']:
            status_parts.append("🔥 SLA Breached")
        if t['escalated']:
            status_parts.append(" escalated")
        
        status = ", ".join(status_parts) if status_parts else "Active"

        print(
            f"  - ID: {t['ticket_id']} ({t['customer_tier']}) | Sentiment: {t['sentiment_score']:.2f} | "
            f"SLA: {t['sla_steps_remaining']} steps | Status: {status}"
        )
        print(f"    Subject: {t['subject']}")
        print(f"    Body: {t['body']}")
        print(f"    Interactions: {t['interaction_count']}, Sentiment History: {[round(s, 2) for s in t['sentiment_history']]}")
    print("="*50)


def get_action_from_user() -> dict:
    """Prompts the user to enter an action JSON."""
    print("\nEnter your action as a JSON object.")
    print('Example: {"action_type": "respond", "parameters": {"ticket_id": "t1", "tone": "empathetic"}}')
    
    while True:
        try:
            action_str = input("Your Action > ")
            action = json.loads(action_str)
            if "action_type" in action and "parameters" in action:
                return action
            else:
                print("Invalid format. JSON must have 'action_type' and 'parameters' keys.")
        except json.JSONDecodeError:
            print("Invalid JSON. Please check your syntax.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session.")
            return None


def main():
    """Main function to run the interactive CLI."""
    print("--- SupportSentinel Interactive CLI ---")

    # 1. Choose a task
    try:
        tasks_response = requests.get(f"{API_BASE_URL}/tasks")
        tasks_response.raise_for_status()
        tasks = tasks_response.json()
    except requests.RequestException as e:
        print(f"\n[ERROR] Could not connect to the environment server at {API_BASE_URL}.")
        print("Please ensure the server is running: uvicorn customer-support-env.app:app --port 7860")
        return

    print("\nAvailable tasks:")
    for i, (task_id, details) in enumerate(tasks.items()):
        print(f"  {i+1}. {task_id} ({details['difficulty']}) - {details['description']}")

    while True:
        try:
            choice = int(input("\nChoose a task number to start: "))
            if 1 <= choice <= len(tasks):
                task_id = list(tasks.keys())[choice - 1]
                break
            else:
                print("Invalid number.")
        except ValueError:
            print("Please enter a number.")

    # 2. Reset the environment
    seed = 42
    try:
        response = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_id": task_id, "seed": seed}
        )
        response.raise_for_status()
        observation = response.json()
        session_id = response.headers['X-Session-Id']
        print(f"\nStarted new session for task '{task_id}'. Session ID: {session_id}")
    except requests.RequestException as e:
        print(f"[ERROR] Failed to reset environment: {e}")
        return

    done = False
    while not done:
        print_observation(observation)
        
        action = get_action_from_user()
        if action is None:
            break

        # 3. Take a step
        try:
            response = requests.post(
                f"{API_BASE_URL}/step?session_id={session_id}",
                json=action
            )
            response.raise_for_status()
            step_data = response.json()
        except requests.RequestException as e:
            print(f"\n[ERROR] Failed to step environment: {e.response.json()['detail']}")
            continue

        observation = step_data['observation']
        reward_info = step_data['reward']
        done = step_data['done']

        print("\n--- Step Result ---")
        print(f"Feedback: {reward_info['feedback']}")
        print(f"Step Score: {reward_info['score']:.2f}")
        print(f"Cumulative Score: {reward_info['cumulative_score']:.2f}")
        print("-------------------")

    print("\n--- Episode Finished ---")
    final_score = observation['current_score']
    print(f"Final Score: {final_score:.2f}")
    print(f"Total Steps: {observation['step_number']}")


if __name__ == "__main__":
    main()
