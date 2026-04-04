"""
Pydantic models for the SupportSentinelEnv environment.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    """Represents a single customer support ticket."""
    ticket_id: str
    subject: str
    body: str
    customer_name: str
    customer_tier: str = Field(..., pattern="^(free|pro|enterprise)$")
    category: str = Field(..., pattern="^(billing|technical|account|shipping|general)$")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sla_steps_remaining: int
    sla_total_steps: int
    interaction_count: int = 0
    sentiment_history: List[float] = []
    resolved: bool = False
    escalated: bool = False
    sla_breached: bool = False
    value: float = 1.0
    status: str = "pending"

    def model_post_init(self, __context: Any) -> None:
        """Ensure sentiment history starts with the initial score."""
        if not self.sentiment_history:
            self.sentiment_history.append(self.sentiment_score)


class Action(BaseModel):
    """Represents an action taken by the agent."""
    action_type: str = Field(..., pattern="^(prioritize|respond|escalate|compensate|resolve|defer)$")
    parameters: Dict[str, Any]


class Observation(BaseModel):
    """Represents the observation returned to the agent."""
    tickets: List[Ticket]
    task_id: str
    task_description: str
    step_number: int
    max_steps: int
    available_actions: List[str]
    current_score: float


class Reward(BaseModel):
    """Represents the reward for a step."""
    score: float
    partial_scores: Dict[str, float]
    feedback: str
    cumulative_score: float


class EnvState(BaseModel):
    """Represents the full state of the environment for a session."""
    session_id: str
    task_id: str
    step_number: int
    max_steps: int
    done: bool
    cumulative_score: float
    tickets: List[Ticket]


class StepResponse(BaseModel):
    """The combined response for a `step` action."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


class ResetRequest(BaseModel):
    """
    Model for the optional body of the /reset endpoint.
    """
    task_id: Optional[str] = "sla_triage"
    session_id: Optional[str] = None
    seed: Optional[int] = 42
