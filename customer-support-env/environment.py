"""
Core environment logic for the SupportSentinelEnv.
"""
import copy
import random
from typing import List, Dict, Any, Tuple

try:
    from .models import Ticket, Action, Observation, Reward, EnvState
    from .tasks import TASK_DEFINITIONS
    from .graders import GRADER_FUNCTIONS
except ImportError:
    from models import Ticket, Action, Observation, Reward, EnvState
    from tasks import TASK_DEFINITIONS
    from graders import GRADER_FUNCTIONS

# Deterministic sentiment change values
SENTIMENT_CHANGE_TABLE = {
    "respond": {
        "empathetic": 0.20,
        "apologetic": 0.25,
        "solution_focused": 0.15,
        "formal": 0.05,
    },
    "escalate": {
        "correct": 0.10,
        "wrong": -0.10,
    },
    "compensate": {
        "credit": 0.25,
        "refund": 0.35,
        "priority_upgrade": 0.20,
        "free_month": 0.30,
    },
    "resolve": {
        "positive_sentiment_bonus": 0.10,
        "negative_sentiment_penalty": -0.05,
    },
    "defer": -0.10,
    "action_on_resolved_penalty": -0.05,
}

class SupportSentinelEnv:
    """
    An OpenEnv-compliant environment for an AI customer support agent.
    """

    def __init__(self, task_id: str, seed: int = 42):
        """
        Initializes the environment for a specific task.

        Args:
            task_id: The identifier for the task to load.
            seed: A seed for the random number generator for reproducibility.
        """
        if task_id not in TASK_DEFINITIONS:
            raise ValueError(f"Task with id '{task_id}' not found.")

        self.task_id = task_id
        self.task_def = TASK_DEFINITIONS[task_id]
        self.seed = seed
        self.random = random.Random(seed)

        self.tickets: List[Ticket] = []
        self.step_number = 0
        self.max_steps = self.task_def["max_steps"]
        self.cumulative_score = 0.01  # Start at minimum valid score (0 < 0.01 < 1)
        self.done = False

        self._reset_internal_state()

    def _reset_internal_state(self):
        """Resets the environment to its initial state for the current task."""
        self.tickets = [t.copy(deep=True) for t in self.task_def["tickets"]]
        self.step_number = 0
        self.cumulative_score = 0.01  # Reset to minimum valid score
        self.done = False

    def reset(self) -> Observation:
        """
        Resets the environment and returns the initial observation.

        Returns:
            The initial observation of the environment.
        """
        self._reset_internal_state()
        return self._get_observation()

    def _get_observation(self) -> Observation:
        """
        Constructs the current observation of the environment.

        Returns:
            The current observation.
        """
        return Observation(
            tickets=[t.copy(deep=True) for t in self.tickets],
            task_id=self.task_id,
            task_description=self.task_def["description"],
            step_number=self.step_number,
            max_steps=self.max_steps,
            available_actions=self.task_def["available_actions"],
            current_score=self.cumulative_score,
        )

    def _apply_sentiment_change(self, ticket: Ticket, change: float, feedback_log: list):
        """Applies a sentiment change to a ticket, capping at [-1.0, 1.0]."""
        initial_score = ticket.sentiment_score
        ticket.sentiment_score = max(-1.0, min(1.0, ticket.sentiment_score + change))
        ticket.sentiment_history.append(ticket.sentiment_score)
        feedback_log.append(f"Ticket {ticket.ticket_id} sentiment changed from {initial_score:.2f} to {ticket.sentiment_score:.2f} (change: {change:.2f}).")

    def _update_slas(self, feedback_log: list = None):
        """Updates the SLA counters for all unresolved tickets and applies sentiment decay.
        
        Sentiment decay represents growing customer frustration when their tickets are ignored.
        Each unresolved ticket loses 0.05 sentiment per step they remain unresolved.
        """
        if feedback_log is None:
            feedback_log = []
        
        for ticket in self.tickets:
            if not ticket.resolved:
                # SLA countdown
                if ticket.sla_steps_remaining > 0:
                    ticket.sla_steps_remaining -= 1
                if ticket.sla_steps_remaining == 0 and not ticket.sla_breached:
                    ticket.sla_breached = True
                    # Apply a one-time sentiment penalty for breaching SLA
                    self._apply_sentiment_change(ticket, -0.25, feedback_log)
                
                # ADVANCED DYNAMIC: Sentiment decay (customers get angrier if ignored)
                # This applies EVERY step an unresolved ticket is NOT being handled
                self._apply_sentiment_change(ticket, -0.05, feedback_log)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Executes a single step in the environment.

        Args:
            action: The action to be performed by the agent.

        Returns:
            A tuple containing the new observation, the reward, a done flag, and an info dict.
        """
        if self.done:
            raise RuntimeError("Cannot step in a completed episode. Please reset the environment.")

        self.step_number += 1
        initial_tickets_state = [t.copy(deep=True) for t in self.tickets]
        feedback_log = []

        # --- 1. Apply Action ---
        try:
            self._execute_action(action, feedback_log)
        except (ValueError, KeyError, TypeError) as e:
            # Invalid action format or parameters
            self.done = True # End episode on invalid action
            reward = Reward(
                score=0.01,
                partial_scores={"validation_error": 0.01},
                feedback=f"Invalid action: {e}. Episode terminated.",
                cumulative_score=max(0.01, min(0.99, self.cumulative_score))
            )
            return self._get_observation(), reward, self.done, {}

        # --- 2. Update Environment State (SLAs and Sentiment Decay) ---
        # For sla_triage, SLA is handled by the grader, not per step.
        if self.task_id != "sla_triage":
            self._update_slas(feedback_log)

        # --- 3. Check for Termination Conditions ---
        if self.step_number >= self.max_steps:
            self.done = True
            feedback_log.append("Max steps reached.")
        
        if all(t.resolved for t in self.tickets):
            self.done = True
            feedback_log.append("All tickets resolved.")
        
        # In sla_triage, the episode ends immediately after the first action.
        if self.task_id == "sla_triage":
            self.done = True

        # --- 4. Calculate Reward ---
        grader = GRADER_FUNCTIONS[self.task_id]
        
        # Graders for multi-step tasks need to know if it's a final or intermediate step.
        if self.task_id == "sla_triage":
            reward = grader(action.dict(), initial_tickets_state, self.tickets, self.cumulative_score)
        else:
            reward = grader(action.dict(), initial_tickets_state, self.tickets, self.cumulative_score, self.done, self.max_steps)

        # CRITICAL: Add defensive clamping to reward before use
        # This ensures score and cumulative_score are strictly in (0.01, 0.99)
        if reward:
            reward.score = max(0.01, min(0.99, reward.score))
            reward.cumulative_score = max(0.01, min(0.99, reward.cumulative_score))
        
        # For multi-step tasks, the final cumulative score is the final reward.
        # For single-step tasks, we add the score to the (zero) base.
        # CRITICAL: Clamp cumulative_score to stay within (0, 1) for framework validation
        if self.done and self.task_id != "sla_triage":
             self.cumulative_score = max(0.01, min(0.99, reward.cumulative_score))
        elif self.task_id == "sla_triage":
             self.cumulative_score = max(0.01, min(0.99, self.cumulative_score + reward.score))
        # On intermediate steps, cumulative_score is handled by the grader logic.
        
        # --- 5. Return results ---
        observation = self._get_observation()
        info = {"feedback_log": feedback_log}

        return observation, reward, self.done, info

    def _execute_action(self, action: Action, feedback_log: list):
        """Internal logic to execute a given action and modify the state."""
        action_type = action.action_type
        params = action.parameters

        if action_type not in self.task_def["available_actions"]:
            raise ValueError(f"Action '{action_type}' is not available for task '{self.task_id}'.")

        if action_type == "prioritize":
            # No state change, handled by grader
            feedback_log.append("`prioritize` action received. Grading will be based on this order.")
            return

        # All other actions require a ticket_id
        ticket_id = params.get("ticket_id")
        if not ticket_id:
            raise ValueError("`ticket_id` is missing from action parameters.")

        try:
            ticket = next(t for t in self.tickets if t.ticket_id == ticket_id)
        except StopIteration:
            raise ValueError(f"Ticket with id '{ticket_id}' not found.")

        if ticket.resolved:
            self._apply_sentiment_change(ticket, SENTIMENT_CHANGE_TABLE["action_on_resolved_penalty"], feedback_log)
            feedback_log.append(f"Action on already resolved ticket {ticket_id} incurred a penalty.")
            return

        ticket.interaction_count += 1

        if action_type == "respond":
            tone = params.get("tone")
            if tone not in SENTIMENT_CHANGE_TABLE["respond"]:
                raise ValueError(f"Invalid tone '{tone}'.")
            change = SENTIMENT_CHANGE_TABLE["respond"][tone]
            self._apply_sentiment_change(ticket, change, feedback_log)

        elif action_type == "escalate":
            department = params.get("department")
            # Correct escalation if department matches ticket category
            if department == ticket.category:
                change = SENTIMENT_CHANGE_TABLE["escalate"]["correct"]
            else:
                change = SENTIMENT_CHANGE_TABLE["escalate"]["wrong"]
            self._apply_sentiment_change(ticket, change, feedback_log)
            ticket.escalated = True

        elif action_type == "compensate":
            comp_type = params.get("type")
            if comp_type not in SENTIMENT_CHANGE_TABLE["compensate"]:
                raise ValueError(f"Invalid compensation type '{comp_type}'.")
            change = SENTIMENT_CHANGE_TABLE["compensate"][comp_type]
            self._apply_sentiment_change(ticket, change, feedback_log)

        elif action_type == "resolve":
            ticket.resolved = True
            if ticket.sentiment_score > 0:
                change = SENTIMENT_CHANGE_TABLE["resolve"]["positive_sentiment_bonus"]
            else:
                change = SENTIMENT_CHANGE_TABLE["resolve"]["negative_sentiment_penalty"]
            self._apply_sentiment_change(ticket, change, feedback_log)
            feedback_log.append(f"Ticket {ticket_id} marked as resolved.")

        elif action_type == "defer":
            change = SENTIMENT_CHANGE_TABLE["defer"]
            self._apply_sentiment_change(ticket, change, feedback_log)

    def get_state(self, session_id: str) -> EnvState:
        """
        Returns the complete current state of the environment for a session.
        """
        return EnvState(
            session_id=session_id,
            task_id=self.task_id,
            step_number=self.step_number,
            max_steps=self.max_steps,
            done=self.done,
            cumulative_score=self.cumulative_score,
            tickets=[t.copy(deep=True) for t in self.tickets],
        )
