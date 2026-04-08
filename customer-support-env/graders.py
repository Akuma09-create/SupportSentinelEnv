"""
Deterministic grading logic for SupportSentinelEnv tasks.
"""
from typing import List, Dict, Any
try:
    from .models import Ticket, Reward
except (ImportError, ValueError):
    from models import Ticket, Reward

def grade_sla_triage(action: Dict[str, Any], initial_tickets: List[Ticket], final_tickets: List[Ticket], cumulative_score: float) -> Reward:
    """
    Grades the 'sla_triage' task.
    Score = % of tickets that would be resolved within SLA given the prioritized order.
    Assumes each ticket resolution takes one step.
    """
    feedback_lines = []
    partial_scores = {}

    try:
        prioritized_ids = action['parameters']['ticket_ids']
        if not isinstance(prioritized_ids, list) or len(prioritized_ids) != len(initial_tickets):
            raise ValueError("Invalid 'ticket_ids' format.")
    except (KeyError, TypeError, ValueError) as e:
        return Reward(
            score=0.01,
            partial_scores={"validation_error": 0.01},
            feedback=f"Invalid action format: {e}",
            cumulative_score=cumulative_score
        )

    ticket_map = {t.ticket_id: t for t in initial_tickets}
    
    # Check for duplicate or missing IDs
    if len(set(prioritized_ids)) != len(initial_tickets) or set(prioritized_ids) != set(ticket_map.keys()):
        return Reward(
            score=0.01,
            partial_scores={"validation_error": 0.01},
            feedback="Error: ticket_ids must contain all unique ticket IDs exactly once.",
            cumulative_score=cumulative_score
        )

    steps_elapsed = 0
    resolved_within_sla = 0
    total_tickets = len(initial_tickets)

    for ticket_id in prioritized_ids:
        steps_elapsed += 1
        ticket = ticket_map[ticket_id]
        if steps_elapsed <= ticket.sla_steps_remaining:
            resolved_within_sla += 1
            feedback_lines.append(f"SUCCESS: Ticket {ticket_id} (SLA: {ticket.sla_steps_remaining}) resolved at step {steps_elapsed}.")
        else:
            feedback_lines.append(f"FAILURE: Ticket {ticket_id} (SLA: {ticket.sla_steps_remaining}) breached at step {steps_elapsed}.")

    score = resolved_within_sla / total_tickets if total_tickets > 0 else 0.01
    # Clamp score to valid range (0, 1) - strictly between 0 and 1
    score = max(0.01, min(0.99, score))
    partial_scores["sla_compliance"] = score
    
    feedback = "SLA Triage Result:\n" + "\n".join(feedback_lines)
    feedback += f"\nFinal Score: {resolved_within_sla}/{total_tickets} tickets met their SLA."

    return Reward(
        score=score,
        partial_scores=partial_scores,
        feedback=feedback,
        cumulative_score=cumulative_score + score
    )


def grade_sentiment_recovery(action: Dict[str, Any], initial_tickets: List[Ticket], final_tickets: List[Ticket], cumulative_score: float, done: bool, max_steps: int) -> Reward:
    """
    Grades the 'sentiment_recovery' task.
    On final step: Score = (final_sentiment + 1) / 2 * sla_bonus + resolution_bonus.
    On intermediate steps: Provides a small reward based on sentiment change.
    """
    initial_ticket = initial_tickets[0]
    final_ticket = final_tickets[0]

    # --- Step-level reward (if not done) ---
    if not done:
        sentiment_change = final_ticket.sentiment_score - initial_ticket.sentiment_score
        step_score = 0.5 + (sentiment_change * 0.5)  # Nudge around a neutral 0.5
        return Reward(
            score=step_score,
            partial_scores={"sentiment_change": sentiment_change},
            feedback=f"Sentiment changed by {sentiment_change:.2f}.",
            # For this task, cumulative score is only meaningful at the end.
            # We just pass the previous score through.
            cumulative_score=cumulative_score
        )

    # --- Final episode score calculation (if done) ---
    final_sentiment = final_ticket.sentiment_score
    sentiment_score_component = (final_sentiment + 1) / 2.0

    if final_ticket.sla_breached:
        sla_bonus = 0.5
        feedback = f"SLA breached. Final sentiment was {final_sentiment:.2f}."
    else:
        sla_bonus = final_ticket.sla_steps_remaining / final_ticket.sla_total_steps if final_ticket.sla_total_steps > 0 else 0
        feedback = f"Task ended with {final_ticket.sla_steps_remaining} steps remaining. Final sentiment: {final_sentiment:.2f}."

    # Base score from sentiment and SLA
    score = sentiment_score_component * sla_bonus

    # Bonus for resolving the ticket
    resolution_bonus = 0.0
    if final_ticket.status == "resolved":
        resolution_bonus = 0.2 # Give a 20% bonus for resolving
        feedback += " Ticket resolved, adding bonus."
    
    final_score = score + resolution_bonus
    
    # Clamp score to valid range (0, 1) - strictly between 0 and 1
    final_score = max(0.01, min(0.99, final_score))

    partial_scores = {
        "sentiment_component": sentiment_score_component,
        "sla_bonus": sla_bonus,
        "resolution_bonus": resolution_bonus
    }

    # When done, the reward's 'score' and 'cumulative_score' should both be the final calculated score.
    return Reward(
        score=final_score,
        partial_scores=partial_scores,
        feedback=feedback,
        cumulative_score=cumulative_score + final_score
    )


def grade_queue_optimization(action: Dict[str, Any], initial_tickets: List[Ticket], final_tickets: List[Ticket], cumulative_score: float, done: bool, max_steps: int) -> Reward:
    """
    Grades the 'queue_optimization' task.
    A reward is given at each step a ticket is successfully resolved.
    The reward is the ticket's value. The final score is the sum of rewards.
    """
    # 1. Check if the action was to resolve a ticket
    if action.get("action_type") != "resolve":
        return Reward(score=0.01, partial_scores={}, feedback="No resolve action taken.", cumulative_score=cumulative_score)

    ticket_id_to_resolve = action.get("parameters", {}).get("ticket_id")
    if not ticket_id_to_resolve:
        return Reward(score=0.01, partial_scores={}, feedback="Resolve action taken, but no ticket_id specified.", cumulative_score=cumulative_score)

    # 2. Find the ticket in the state *before* the action to get its value
    initial_ticket = next((t for t in initial_tickets if t.ticket_id == ticket_id_to_resolve), None)
    if not initial_ticket:
        return Reward(score=0.01, partial_scores={}, feedback=f"Agent tried to resolve ticket {ticket_id_to_resolve}, but it wasn't in the initial state for this step.", cumulative_score=cumulative_score)

    # 3. Find the ticket in the state *after* the action to confirm it was resolved
    final_ticket = next((t for t in final_tickets if t.ticket_id == ticket_id_to_resolve), None)
    if not final_ticket or not final_ticket.resolved:
        return Reward(
            score=0.01,
            partial_scores={},
            feedback=f"Action to resolve ticket {ticket_id_to_resolve} was taken, but it was not resolved in the final state.",
            cumulative_score=cumulative_score
        )

    # 4. If resolved, the reward for this step is the ticket's value.
    # Clamp to valid range (0, 1)
    step_reward = max(0.01, min(0.99, initial_ticket.value))
    
    return Reward(
        score=step_reward,
        partial_scores={"resolved_value": initial_ticket.value},
        feedback=f"Successfully resolved ticket {ticket_id_to_resolve} worth {initial_ticket.value:.1f}. Step reward: {step_reward:.1f}",
        cumulative_score=cumulative_score + step_reward
    )


GRADER_FUNCTIONS = {
    "sla_triage": grade_sla_triage,
    "sentiment_recovery": grade_sentiment_recovery,
    "queue_optimization": grade_queue_optimization,
}
