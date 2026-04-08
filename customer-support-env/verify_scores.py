"""
Verification script to test the scoring logic of SupportSentinelEnv tasks.

This script runs predefined "optimal" action sequences for each task
and asserts that the final scores match the expected outcomes. It serves as an
integration test for the environment's core logic and graders.
"""
import os
import requests
import time

# --- Configuration ---
API_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
TOLERANCE = 0.02  # Tolerance for floating point comparisons

def run_test(test_func):
    """Decorator to run a test function and print its status."""
    def wrapper():
        test_name = test_func.__name__
        print(f"--- Running test: {test_name} ---")
        try:
            test_func()
            print(f"[PASS] {test_name}")
        except (AssertionError, requests.RequestException) as e:
            print(f"[FAIL] {test_name}")
            print(f"  Reason: {e}")
        print("-" * (20 + len(test_name)))
        time.sleep(1)
    return wrapper

def post_request(endpoint: str, payload: dict, session_id: str = None) -> tuple:
    """Helper function to make POST requests to the server."""
    url = f"{API_BASE_URL}{endpoint}"
    if session_id:
        url += f"?session_id={session_id}"
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    new_session_id = response.headers.get('X-Session-Id', session_id)
    return response.json(), new_session_id

@run_test
def test_sla_triage():
    """Tests the sla_triage task with the optimal prioritization."""
    # Expected score: 0.99 (5/5 tickets meet SLA, clamped to 0.99)
    expected_score = 0.99
    
    # Reset environment
    _, session_id = post_request("/reset", {"task_id": "sla_triage", "seed": 42})
    
    # Perform the optimal action
    action = {
        "action_type": "prioritize",
        "parameters": {"ticket_ids": ["t1", "t4", "t2", "t5", "t3"]}
    }
    result, _ = post_request("/step", action, session_id)
    
    final_score = result['reward']['score']
    print(f"  Final score: {final_score:.2f} (Expected: {expected_score:.2f})")
    assert abs(final_score - expected_score) < TOLERANCE, f"Score {final_score} not close to {expected_score}"

@run_test
def test_sentiment_recovery():
    """Tests the sentiment_recovery task with the optimal action sequence."""
    # Re-calculated expected score based on the grader's logic:
    # sentiment_component = (0.1 + 1)/2 = 0.55
    # sla_bonus = 2/6 = 0.333
    # base_score = 0.55 * 0.333 = 0.183
    # resolve_bonus = +0.1
    # final_score = 0.183 + 0.1 = ~0.28
    expected_score = 0.28
    
    # Reset environment
    _, session_id = post_request("/reset", {"task_id": "sentiment_recovery", "seed": 42})
    
    # This is the correct sequence to get sentiment positive before resolving.
    actions = [
        {"action_type": "respond", "parameters": {"ticket_id": "t_angry", "tone": "apologetic"}},
        {"action_type": "compensate", "parameters": {"ticket_id": "t_angry", "type": "refund"}},
        {"action_type": "respond", "parameters": {"ticket_id": "t_angry", "tone": "empathetic"}},
        {"action_type": "resolve", "parameters": {"ticket_id": "t_angry", "note": "Resolved."}}
    ]
    
    final_result = None
    for action in actions:
        result, _ = post_request("/step", action, session_id)
        final_result = result
        if result['done']:
            break
            
    final_score = final_result['reward']['cumulative_score']
    print(f"  Final score: {final_score:.2f} (Expected: ~{expected_score:.2f})")
    assert abs(final_score - expected_score) < TOLERANCE, f"Score {final_score} not close to {expected_score}"

import sys
from pathlib import Path

# Add the project root to the Python path
# This allows us to import from the 'models' module within the 'customer-support-env' directory
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import Ticket # Import the Ticket model

# ... (existing code) ...

@run_test
def test_queue_optimization():
    """Tests the queue_optimization task with a simple heuristic (prioritize enterprise)."""
    # Expected score: ~0.65 (This is an estimate, the heuristic is not perfect)
    expected_score = 0.65
    
    # Reset environment
    obs, session_id = post_request("/reset", {"task_id": "queue_optimization", "seed": 42})
    
    final_result = None
    for step in range(15):
        # Parse ticket dictionaries into Ticket objects
        all_tickets = [Ticket(**t_data) for t_data in obs['tickets']]
        
        # Simple heuristic: find the highest priority (enterprise > pro > free) non-resolved ticket
        # with the lowest SLA.
        active_tickets = [t for t in all_tickets if not t.resolved]
        if not active_tickets:
            break
            
        tier_map = {"enterprise": 0, "pro": 1, "free": 2}
        active_tickets.sort(key=lambda t: (tier_map[t.customer_tier], t.sla_steps_remaining))
        
        target_ticket = active_tickets[0]
        
        # Simple action heuristic
        if target_ticket.sentiment_score < -0.5:
            action = {"action_type": "respond", "parameters": {"ticket_id": target_ticket.ticket_id, "tone": "apologetic"}}
        elif target_ticket.sentiment_score < 0:
            action = {"action_type": "respond", "parameters": {"ticket_id": target_ticket.ticket_id, "tone": "empathetic"}}
        else:
            action = {"action_type": "resolve", "parameters": {"ticket_id": target_ticket.ticket_id, "note": "Resolved."}}

        result, _ = post_request("/step", action, session_id)
        obs = result['observation']
        final_result = result
        if result['done']:
            break
            
    final_score = final_result['reward']['cumulative_score']
    print(f"  Final score: {final_score:.2f} (Expected: ~{expected_score:.2f})")
    assert abs(final_score - expected_score) < TOLERANCE, f"Score {final_score} not close to {expected_score}"


def main():
    """Runs all verification tests."""
    print("="*40)
    print(" Starting SupportSentinelEnv Score Verification")
    print("="*40)
    
    # Check server health
    try:
        requests.get(f"{API_BASE_URL}/health").raise_for_status()
        print("Server is running. Proceeding with tests...\n")
    except requests.RequestException:
        print(f"[ERROR] Server not found at {API_BASE_URL}.")
        print("Please start the server before running verification.")
        return
        
    test_sla_triage()
    test_sentiment_recovery()
    test_queue_optimization()
    
    print("\nVerification complete.")
    print("="*40)

if __name__ == "__main__":
    main()
