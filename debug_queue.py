import requests
import json
import time
import subprocess
import os

BASE_URL = "http://localhost:7860"
SESSION_ID = "debug_queue_optimization_v5"
UVICORN_PROCESS = None

def start_server():
    """Starts the Uvicorn server in a separate process."""
    global UVICORN_PROCESS
    print("--- Attempting to start Uvicorn server ---")
    
    python_executable = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
    
    command = [
        python_executable,
        "-m", "uvicorn",
        "customer-support-env.app:app",
        "--host", "0.0.0.0",
        "--port", "7860"
    ]
    
    project_root = os.getcwd()
    
    try:
        server_log_file = open("server_log.txt", "w")
        UVICORN_PROCESS = subprocess.Popen(command, cwd=project_root, stdout=server_log_file, stderr=subprocess.STDOUT)
        print(f"Server process started with PID: {UVICORN_PROCESS.pid}. Logging to server_log.txt")
        time.sleep(5)

        if UVICORN_PROCESS.poll() is not None:
            print("[ERROR] Server process terminated prematurely. Check server_log.txt for details.")
            return False
        
        print("Server appears to be running.")
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
        try:
            UVICORN_PROCESS.wait(timeout=5)
            print("Server process stopped.")
        except subprocess.TimeoutExpired:
            print("Server did not terminate gracefully, killing.")
            UVICORN_PROCESS.kill()
        UVICORN_PROCESS = None

def run_test():
    """
    Runs a targeted test on the 'queue_optimization' task.
    """
    if not start_server():
        return

    try:
        print("\n--- Starting targeted test for: queue_optimization ---")

        # 1. Reset the environment
        reset_payload = {"task_name": "queue_optimization", "session_id": SESSION_ID}
        resp = requests.post(f"{BASE_URL}/reset", json=reset_payload, timeout=15)
        resp.raise_for_status()
        observation = resp.json()
        print("RESET successful.")

        # 2. Perform the 'resolve' action
        ticket_to_resolve = observation["tickets"][0]["ticket_id"]
        action_payload = {"action_type": "resolve", "parameters": {"ticket_id": ticket_to_resolve}}
        print(f"\n--- Performing STEP with action: {json.dumps(action_payload)} ---")
        
        step_url = f"{BASE_URL}/step?session_id={SESSION_ID}"
        resp = requests.post(step_url, json=action_payload, timeout=15)

        print(f"STEP response status code: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[ERROR] Received status {resp.status_code}. Server response:")
            print(resp.text)
        else:
            print("STEP successful.")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] An error occurred during the test: {e}")
    finally:
        stop_server()
        print("\n--- Server Log (server_log.txt) ---")
        try:
            with open("server_log.txt", "r") as f:
                print(f.read())
        except FileNotFoundError:
            print("server_log.txt not found.")

if __name__ == "__main__":
    run_test()
