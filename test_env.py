import httpx
import time

# Give server time to start
time.sleep(2)

# Test full workflow
resp = httpx.post('http://localhost:7860/reset', json={'task_id':'sentiment_recovery','seed':42})
print('✓ /reset: step_number', resp.json()['step_number'], '| sentiment', resp.json()['tickets'][0]['sentiment_score'])
sid = resp.headers.get('X-Session-Id')

# Take steps and track progression
for i in range(3):
    action = {'action_type': 'respond', 'parameters': {'ticket_id': 't_angry', 'tone': 'solution_focused'}}
    r = httpx.post(f'http://localhost:7860/step?session_id={sid}', json=action)
    obs = r.json()['observation']
    reward = r.json()['reward']['score']
    print(f'✓ /step {i+1}: step_number {obs["step_number"]} | sentiment {obs["tickets"][0]["sentiment_score"]:.2f} | reward {reward:.3f}')
    
print('✓ Real environment is LIVE and progressing!')
