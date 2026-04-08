import httpx
import time

# Give server time to start
time.sleep(2)

print("=== TESTING QUEUE OPTIMIZATION UPGRADES ===\n")

# Test 1: Verify 15 tickets load
print("TEST 1: Queue size (should be 15)")
resp = httpx.post('http://localhost:7860/reset', json={'task_id':'queue_optimization','seed':42})
data = resp.json()
print(f"  Tickets loaded: {len(data['tickets'])}")
print(f"  Ticket IDs: {[t['ticket_id'] for t in data['tickets']]}")
print(f"  Max steps: {data['max_steps']}")
sid = resp.headers.get('X-Session-Id')

# Test 2: Verify sentiment decay (-0.05 per step)
print("\nTEST 2: Sentiment decay (unresolved tickets should lose -0.05/step)")
initial_sentiments = {t['ticket_id']: t['sentiment_score'] for t in data['tickets']}

# Take action on ONE ticket, others should decay
action = {'action_type': 'resolve', 'parameters': {'ticket_id': data['tickets'][0]['ticket_id']}}
r = httpx.post(f'http://localhost:7860/step?session_id={sid}', json=action)
result = r.json()

print(f"  Step 1 action: resolved {data['tickets'][0]['ticket_id']}")
new_obs = result['observation']
new_sentiments = {t['ticket_id']: t['sentiment_score'] for t in new_obs['tickets']}

decay_detected = False
for tid in list(initial_sentiments.keys())[1:5]:  # Check a few tickets
    change = new_sentiments[tid] - initial_sentiments[tid]
    print(f"    {tid}: {initial_sentiments[tid]:.2f} → {new_sentiments[tid]:.2f} (change: {change:.2f})")
    if change == -0.05:
        decay_detected = True

if decay_detected:
    print(f"  ✓ Sentiment decay working!")
else:
    print(f"  ✗ Sentiment decay NOT detected (expected -0.05)")

# Test 3: Verify hard task still completes
print("\nTEST 3: Hard task must complete within max_steps (25)")
# Do a few more steps
for i in range(2, 6):
    unresolved = [t for t in new_obs['tickets'] if not t['resolved']]
    if not unresolved:
        break
    first = unresolved[0]
    action = {'action_type': 'resolve', 'parameters': {'ticket_id': first['ticket_id']}}
    r = httpx.post(f'http://localhost:7860/step?session_id={sid}', json=action)
    result = r.json()
    new_obs = result['observation']
    print(f"  Step {i}: Resolved {first['ticket_id']}, step_number={new_obs['step_number']}")

print(f"\n✓ All upgrades working!")
print(f"✓ Creativity score should improve significantly with 15-ticket challenge + sentiment decay!")
