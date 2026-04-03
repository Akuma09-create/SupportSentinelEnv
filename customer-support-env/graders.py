"""
Deterministic grading logic for SupportSentinelEnv tasks.
"""
from typing import List, Dict, Any
from .models import Ticket, Reward

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
            score=0.0,
            partial_scores={"validation_error": 0.0},
            feedback=f"Invalid action format: {e}",
            cumulative_score=cumulative_score
        )

    ticket_map = {t.ticket_id: t for t in initial_tickets}
    
    # Check for duplicate or missing IDs
    if len(set(prioritized_ids)) != len(initial_tickets) or set(prioritized_ids) != set(ticket_map.keys()):
        return Reward(
            score=0.0,
            partial_scores={"validation_error": 0.0},
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

    score = resolved_within_sla / total_tickets if total_tickets > 0 else 0.0
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
    
    # Ensure score is capped at 1.0
    final_score = min(final_score, 1.0)

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


def grade_queue_optimization(action: Dict[str, Any], initial_tickets: List[Ticket], final_tickets: List[Ticket], cumulative_score: float, done: bool) -> Reward:
    """
    Grades the 'queue_optimization' task.
    Score is based on the value of tickets resolved.
    A bonus is given for resolving tickets with higher value first.
    """
    # This grader is called at the end of the episode.
    # The action that leads to the 'done' state is the final 'resolve'.
    if not done:
        # Intermediate steps have no reward, but we must return a valid Reward object.
        return Reward(
            score=0.0,
            partial_scores={},
            feedback="Ticket resolved. Awaiting end of episode for final score.",
            cumulative_score=cumulative_score
        )

    # --- Final score calculation ---
    resolved_tickets = [t for t in final_tickets if t.status == "resolved"]
    
    if not resolved_tickets:
        return Reward(
            score=0.0,
            partial_scores={"total_value": 0},
            feedback="No tickets were resolved.",
            cumulative_score=cumulative_score
        )

    total_value = sum(t.value for t in resolved_tickets)
    
    # Normalize the score against the total possible value from the initial set of tickets
    score = total_value / max_possible_value if max_possible_value > 0 else 0

    return Reward(
        score=score,
        partial_scores={"total_value": total_value},
        feedback=f"Episode finished. Total value of resolved tickets: {total_value}",
        cumulative_score=cumulative_score + score
    )


GRADER_FUNCTIONS = {
    "sla_triage": grade_sla_triage,
    "sentiment_recovery": grade_sentiment_recovery,
    "queue_optimization": grade_queue_optimization,
}
