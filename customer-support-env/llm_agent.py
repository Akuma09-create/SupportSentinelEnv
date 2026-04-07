"""
OpenEnv-compliant LLM Agent for SupportSentinelEnv.
This agent uses the OpenAI client to make intelligent decisions about customer support tickets.
Strictly follows the pre-submission checklist requirements.
"""
import os
import json
import sys
from typing import Optional, Dict, Any, List
from openai import OpenAI
import httpx

# ============================================================================
# ENVIRONMENT VARIABLES - As per hackathon requirements
# ============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")  # NO default - required

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Use placeholder if HF_TOKEN not provided (for local testing)
if not HF_TOKEN:
    HF_TOKEN = "test-token-local"

# ============================================================================
# OpenAI Client Configuration - Using environment variables
# ============================================================================
client = OpenAI(api_key=HF_TOKEN)

# ============================================================================
# System Prompts for Each Task
# ============================================================================
SYSTEM_PROMPTS = {
    "sla_triage": """You are an expert customer support dispatcher.
Your task: prioritize customer tickets based on their SLA deadlines and sentiment.
Return ONLY valid JSON with no explanation.
Format: {"action_type": "prioritize", "parameters": {"ticket_ids": [...]}}""",

    "sentiment_recovery": """You are an expert customer relations specialist.
Your task: improve customer satisfaction from initial negative state to +0.3 or higher.
Strategy:
- If sentiment < -0.6: ESCALATE (show expertise)
- If sentiment -0.6 to -0.3: COMPENSATE (show empathy with solutions)
- If sentiment -0.3 to 0.2: RESPOND with solution-focused tone
- If sentiment near target: RESOLVE to finalize
Return ONLY valid JSON with no explanation.
Example: {"action_type": "compensate", "parameters": {"ticket_id": "t_angry", "type": "refund"}}""",

    "queue_optimization": """You are an expert support operations manager.
Your task: resolve tickets efficiently to optimize the support queue and maximize value.
Prioritize high-value enterprise tickets first. Each resolved ticket adds its value to your score.
Return ONLY valid JSON with no explanation.
Format: {"action_type": "resolve", "parameters": {"ticket_id": "..."}}"""
}

# ============================================================================
# HTTP Client for Environment Interaction
# ============================================================================
http_client = httpx.Client(base_url=API_BASE_URL, timeout=30.0)

# ============================================================================
# Structured Logging Functions - START/STEP/END format (exactly as required)
# ============================================================================
def log_start(task_id: str, env_name: str, model: str) -> None:
    """Log episode start in required format."""
    print(f"START task={task_id} env={env_name} model={model}", flush=True)

def log_step(step_num: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Log individual step in required format."""
    error_str = f"error={error}" if error else "error=null"
    print(f"STEP step={step_num} action={action} reward={reward:.3f} done={str(done).lower()} {error_str}", flush=True)

def log_end(success: bool, total_steps: int, final_score: float, rewards_list: List[float]) -> None:
    """Log episode end in required format."""
    rewards_str = ",".join(f"{r:.3f}" for r in rewards_list)
    print(f"END success={str(success).lower()} steps={total_steps} score={final_score:.3f} rewards=[{rewards_str}]", flush=True)

# ============================================================================
# LLM Agent Class
# ============================================================================
class LLMAgent:
    """An OpenEnv-compliant agent powered by LLM using OpenAI client."""
    
    def __init__(self, task_id: str, base_url: str = API_BASE_URL, model: str = MODEL_NAME):
        self.task_id = task_id
        self.base_url = base_url
        self.model = model
        self.session_id = None
        self.step_count = 0
        self.rewards = []
        self.current_observation = None
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and get initial observation."""
        try:
            response = http_client.post(
                "/reset",
                json={"task_id": self.task_id}
            )
            response.raise_for_status()
            data = response.json()
            self.session_id = data.get("session_id")
            self.current_observation = data.get("environment", {})
            self.step_count = 0
            self.rewards = []
            return self.current_observation
        except Exception as e:
            print(f"[ERROR] Failed to reset environment: {e}", file=sys.stderr)
            raise
    
    def get_llm_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI client to decide the next action based on observation."""
        user_prompt = self._format_observation(observation)
        system_prompt = SYSTEM_PROMPTS.get(self.task_id, "You are a helpful AI agent.")
        
        try:
            # All LLM calls use the OpenAI client configured via environment variables
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            action_text = response.choices[0].message.content.strip()
            
            # Parse LLM response (expect JSON)
            try:
                action = json.loads(action_text)
                return action
            except json.JSONDecodeError:
                print(f"[WARNING] LLM returned non-JSON: {action_text}", file=sys.stderr)
                return self._get_fallback_action(observation)
                
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
            return self._get_fallback_action(observation)
    
    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation into a readable prompt for the LLM."""
        task_desc = obs.get("task_description", "")
        step_info = f"Step {obs.get('step_number', 0)}/{obs.get('max_steps', 0)}"
        
        tickets_info = "Active tickets:\n"
        for ticket in obs.get("tickets", []):
            if not ticket.get("resolved", False):
                tickets_info += (
                    f"- {ticket.get('ticket_id', 'N/A')}: "
                    f"Sentiment={ticket.get('sentiment_score', 0):.2f}, "
                    f"SLA={ticket.get('sla_steps_remaining', 0)} steps, "
                    f"Tier={ticket.get('customer_tier', 'N/A')}\n"
                )
        
        actions = ", ".join(obs.get("available_actions", []))
        
        return f"""Task: {task_desc}
{step_info}
Available Actions: {actions}

{tickets_info}

What action should we take next?"""
    
    def _get_fallback_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback action if LLM call fails - uses intelligent strategy."""
        if self.task_id == "sla_triage":
            ticket_ids = [t["ticket_id"] for t in observation.get("tickets", [])]
            return {"action_type": "prioritize", "parameters": {"ticket_ids": ticket_ids}}
        
        elif self.task_id == "sentiment_recovery":
            tickets = observation.get("tickets", [])
            unresolved = [t for t in tickets if not t.get("resolved")]
            
            if not unresolved:
                return {"action_type": "defer", "parameters": {}}
            
            ticket = unresolved[0]
            sentiment = ticket.get("sentiment_score", 0)
            step = observation.get("step_number", 0)
            max_steps = observation.get("max_steps", 8)
            steps_left = max_steps - step
            
            # Adaptive strategy based on sentiment and steps remaining
            if sentiment < -0.6:  # Very angry
                # Escalate first for very upset customers
                return {"action_type": "escalate", "parameters": {"ticket_id": ticket["ticket_id"]}}
            elif sentiment < -0.3:  # Angry
                # Compensate to show willingness to resolve
                return {"action_type": "compensate", "parameters": {"ticket_id": ticket["ticket_id"], "type": "refund"}}
            elif sentiment < 0.2:  # Unhappy but recoverable
                # Respond with right tone to build rapport
                return {"action_type": "respond", "parameters": {"ticket_id": ticket["ticket_id"], "tone": "solution_focused"}}
            elif steps_left <= 2 and sentiment < 0.3:  # Near end and not at target
                # Final push with compensation
                return {"action_type": "compensate", "parameters": {"ticket_id": ticket["ticket_id"], "type": "priority_support"}}
            else:  # Good sentiment or time to resolve
                # Resolve when sentiment is good enough
                return {"action_type": "resolve", "parameters": {"ticket_id": ticket["ticket_id"]}}
        
        elif self.task_id == "queue_optimization":
            tickets = observation.get("tickets", [])
            unresolved = [t for t in tickets if not t.get("resolved")]
            if unresolved:
                # Prioritize high-value tickets
                unresolved.sort(key=lambda t: t.get("value", 0), reverse=True)
                return {"action_type": "resolve", "parameters": {"ticket_id": unresolved[0]["ticket_id"]}}
        
        return {"action_type": "defer", "parameters": {"ticket_id": observation.get("tickets", [{}])[0].get("ticket_id", "")}}
    
    def step(self, action: Dict[str, Any]) -> tuple:
        """Take a step in the environment."""
        try:
            response = http_client.post(
                "/step",
                json={"session_id": self.session_id, "action": action}
            )
            response.raise_for_status()
            data = response.json()
            
            observation = data.get("environment", {})
            reward_data = data.get("reward", {})
            done = data.get("done", False)
            
            self.current_observation = observation
            self.step_count += 1
            reward_value = reward_data.get("score", 0.0)
            self.rewards.append(reward_value)
            
            return observation, reward_value, done, data.get("info", {})
        except Exception as e:
            print(f"[ERROR] Step failed: {e}", file=sys.stderr)
            raise
    
    def run_episode(self) -> Dict[str, Any]:
        """Run a complete episode with structured logging."""
        log_start(self.task_id, "SupportSentinelEnv", self.model)
        
        try:
            observation = self.reset()
            done = False
            
            while not done:
                # Get action from LLM
                action = self.get_llm_action(observation)
                
                # Format action for logging
                action_str = json.dumps(action)
                
                # Take step in environment
                observation, reward, done, info = self.step(action)
                
                # Log step (STEP format required)
                log_step(self.step_count, action_str, reward, done)
            
            # Episode complete - calculate final score
            final_score = sum(self.rewards) if self.rewards else 0.0
            success = final_score > 0.5  # Success threshold
            
            # Log end (END format required)
            log_end(success, self.step_count, final_score, self.rewards)
            
            return {
                "task_id": self.task_id,
                "steps": self.step_count,
                "score": final_score,
                "rewards": self.rewards,
                "success": success
            }
        except Exception as e:
            print(f"[ERROR] Episode failed: {e}", file=sys.stderr)
            log_end(False, self.step_count, 0.0, self.rewards)
            raise

# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Run the LLM agent on all three tasks."""
    tasks = ["sla_triage", "sentiment_recovery", "queue_optimization"]
    
    for task_id in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        try:
            agent = LLMAgent(task_id, model=MODEL_NAME)
            result = agent.run_episode()
            print(f"\nTask {task_id} completed: {result}\n", flush=True)
        except Exception as e:
            print(f"\nTask {task_id} failed: {e}\n", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
