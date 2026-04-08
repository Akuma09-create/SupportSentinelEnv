"""
Pydantic models for the SupportSentinelEnv environment.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


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
    model_config = ConfigDict(validate_assignment=True)
    
    score: float
    partial_scores: Dict[str, float]
    feedback: str
    cumulative_score: float
    
    @field_validator('score')
    @classmethod
    def validate_score_range(cls, v):
        """Ensure score is strictly between 0 and 1 (exclusive on both ends)."""
        if not (0 < v < 1):
            v = max(0.01, min(0.99, v))
        return v
    
    @field_validator('score', mode='after')
    @classmethod
    def finalize_score_precision(cls, v):
        """Final safety: check for floating-point edge cases and round precisely."""
        v = float(v)
        # If within rounding distance of 0/1, push inward
        if v <= 0.001:
            return 0.01
        if v >= 0.999:
            return 0.99
        # Round to 4 decimals to prevent serialization issues
        rounded = round(v, 4)
        # Extra safety after rounding
        return max(0.01, min(0.99, rounded))
    
    @field_validator('partial_scores', mode='before')
    @classmethod
    def validate_partial_scores(cls, v):
        """Ensure all partial_scores values are strictly between 0 and 1."""
        if isinstance(v, dict):
            return {k: max(0.01, min(0.99, float(val))) for k, val in v.items()}
        return v
    
    @field_validator('partial_scores', mode='after')
    @classmethod
    def finalize_partial_precision(cls, v):
        """Final safety: round all partial scores and ensure boundaries."""
        result = {}
        for k, val in v.items():
            fval = float(val)
            # Check for edge cases
            if fval <= 0.001:
                result[k] = 0.01
            elif fval >= 0.999:
                result[k] = 0.99
            else:
                # Round and clamp
                rounded = round(fval, 4)
                result[k] = max(0.01, min(0.99, rounded))
        return result
    
    @field_validator('cumulative_score')
    @classmethod
    def validate_cumulative_score_range(cls, v):
        """Ensure cumulative score is strictly between 0 and 1 (exclusive on both ends)."""
        if not (0 < v < 1):
            v = max(0.01, min(0.99, v))
        return v
    
    @field_validator('cumulative_score', mode='after')
    @classmethod
    def finalize_cumulative_precision(cls, v):
        """Final safety: check for floating-point edge cases and round precisely."""
        v = float(v)
        # If within rounding distance of 0/1, push inward
        if v <= 0.001:
            return 0.01
        if v >= 0.999:
            return 0.99
        # Round to 4 decimals to prevent serialization issues
        rounded = round(v, 4)
        # Extra safety after rounding
        return max(0.01, min(0.99, rounded))


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
