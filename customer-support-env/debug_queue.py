import requests
import json
import time
import subprocess

BASE_URL = "http://localhost:7860"
SESSION_ID = "debug_queue_optimization_v2"
UVICORN_PROCESS = None

def start_server():
    """Starts the Uvicorn server in a separate process."""
    global UVICORN_PROCESS
    print("--- Attempting to start Uvicorn server ---")
    command = [".\\venv\\Scripts\\python.exe", "-m", "uvicorn", "customer-support-env.app:app", "--host", "0.0.0.0", "--port", "7860"]
    try:
        UVICORN_PROCESS = subprocess.Popen(command, cwd="c:\\Users\\asus4\\Desktop\\Hackethon")
        print(f"Server process started with PID: {UVICORN_PROCESS.pid}")
        time.sleep(5) # Give the server a moment to start
        return True
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        return False

def stop_server():
    """Stops the Uvicorn server."""
    global UVICORN_PROCESS
    if UVICORN_PROCESS:
        print(f"--- Stopping Uvicorn server (PID: {UVICORN_PROCESS.pid}) ---")
        UVICORN_PROCESS.terminate()
        UVICORN_PROCESS.wait()
        print("Server process stopped.")

def run_test():
    """
    Runs a targeted test on the 'queue_optimization' task to debug the 500 error.
    """
    if not start_server():
        return

    try:
        print("\n--- Starting targeted test for: queue_optimization ---")

        # 1. Reset the environment
        reset_payload = {
            "task_name": "queue_optimization",
            "session_id": SESSION_ID
        }
        resp = requests.post(f"{BASE_URL}/reset", json=reset_payload, timeout=15)
        resp.raise_for_status()
        observation = resp.json()
        print(f"RESET successful. Initial observation: {json.dumps(observation, indent=2)}")

        # 2. Perform the 'resolve' action
        ticket_to_resolve = observation["tickets"][0]["ticket_id"]
        action_payload = {
            "action_type": "resolve",
            "parameters": {"ticket_id": ticket_to_resolve}
        }
        print(f"\n--- Performing STEP with action: {json.dumps(action_payload)} ---")
        
        step_url = f"{BASE_URL}/step?session_id={SESSION_ID}"
        resp = requests.post(step_url, json=action_payload, timeout=15)

        print(f"STEP response status code: {resp.status_code}")

        if resp.status_code == 500:
            print("[ERROR] Received 500 Internal Server Error.")
            print("Server response body:")
            print(resp.text)
        else:
            resp.raise_for_status()
            step_result = resp.json()
            print(f"STEP successful. Result: {json.dumps(step_result, indent=2)}")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] An error occurred during the test: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        stop_server()

if __name__ == "__main__":
    run_test()
