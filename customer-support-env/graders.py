"""
Deterministic grading logic for SupportSentinelEnv tasks.
All returned scores are kept strictly within the valid submission range.
"""
from typing import List, Dict, Any

try:
    from .models import Ticket, Reward
except (ImportError, ValueError):
    from models import Ticket, Reward


def _clamp_score(value: float, min_val: float = 0.01, max_val: float = 0.99) -> float:
    """Clamp a numeric score to stay within (0, 1) for submission safety."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.01
    return max(min_val, min(max_val, value))


def _clamp_partial(scores: Dict[str, float]) -> Dict[str, float]:
    """Clamp all values in a partial_scores dict to stay within (0, 1)."""
    return {k: _clamp_score(v) for k, v in scores.items()}


def grade_sla_triage(
    action: Dict[str, Any],
    initial_tickets: List[Ticket],
    final_tickets: List[Ticket],
    cumulative_score: float
) -> Reward:
    cumulative_score = _clamp_score(cumulative_score)

    feedback_lines = []

    try:
        prioritized_ids = action["parameters"]["ticket_ids"]
        if not isinstance(prioritized_ids, list) or len(prioritized_ids) != len(initial_tickets):
            raise ValueError("Invalid 'ticket_ids' format.")
    except (KeyError, TypeError, ValueError) as e:
        error_score = _clamp_score(0.01)
        return Reward(
            score=error_score,
            partial_scores=_clamp_partial({"validation_error": 0.01}),
            feedback=f"Invalid action format: {e}",
            cumulative_score=_clamp_score(cumulative_score + error_score),
        )

    ticket_map = {t.ticket_id: t for t in initial_tickets}

    if len(set(prioritized_ids)) != len(initial_tickets) or set(prioritized_ids) != set(ticket_map.keys()):
        error_score = _clamp_score(0.01)
        return Reward(
            score=error_score,
            partial_scores=_clamp_partial({"validation_error": 0.01}),
            feedback="Error: ticket_ids must contain all unique ticket IDs exactly once.",
            cumulative_score=_clamp_score(cumulative_score + error_score),
        )

    steps_elapsed = 0
    resolved_within_sla = 0
    total_tickets = len(initial_tickets)

    for ticket_id in prioritized_ids:
        steps_elapsed += 1
        ticket = ticket_map[ticket_id]
        if steps_elapsed <= ticket.sla_steps_remaining:
            resolved_within_sla += 1
            feedback_lines.append(
                f"SUCCESS: Ticket {ticket_id} (SLA: {ticket.sla_steps_remaining}) resolved at step {steps_elapsed}."
            )
        else:
            feedback_lines.append(
                f"FAILURE: Ticket {ticket_id} (SLA: {ticket.sla_steps_remaining}) breached at step {steps_elapsed}."
            )

    raw_score = resolved_within_sla / total_tickets if total_tickets > 0 else 0.01
    score = _clamp_score(raw_score)

    feedback = "SLA Triage Result:\n" + "\n".join(feedback_lines)
    feedback += f"\nFinal Score: {resolved_within_sla}/{total_tickets} tickets met their SLA."

    return Reward(
        score=score,
        partial_scores=_clamp_partial({"sla_compliance": raw_score}),
        feedback=feedback,
        cumulative_score=_clamp_score(cumulative_score + score),
    )


def grade_sentiment_recovery(
    action: Dict[str, Any],
    initial_tickets: List[Ticket],
    final_tickets: List[Ticket],
    cumulative_score: float,
    done: bool,
    max_steps: int
) -> Reward:
    cumulative_score = _clamp_score(cumulative_score)

    initial_ticket = initial_tickets[0]
    final_ticket = final_tickets[0]

    if not done:
        sentiment_change = final_ticket.sentiment_score - initial_ticket.sentiment_score
        raw_step_score = 0.5 + (sentiment_change * 0.5)
        step_score = _clamp_score(raw_step_score)

        return Reward(
            score=step_score,
            partial_scores=_clamp_partial({"sentiment_change": raw_step_score}),
            feedback=f"Sentiment changed by {sentiment_change:.2f}.",
            cumulative_score=_clamp_score(cumulative_score + step_score),  # accumulate every step
        )

    final_sentiment = final_ticket.sentiment_score
    sentiment_score_component = (final_sentiment + 1) / 2.0

    if final_ticket.sla_breached:
        sla_bonus = 0.5
        feedback = f"SLA breached. Final sentiment was {final_sentiment:.2f}."
    else:
        sla_bonus = (
            final_ticket.sla_steps_remaining / final_ticket.sla_total_steps
            if final_ticket.sla_total_steps > 0 else 0.01
        )
        feedback = (
            f"Task ended with {final_ticket.sla_steps_remaining} steps remaining. "
            f"Final sentiment: {final_sentiment:.2f}."
        )

    is_resolved = final_ticket.status == "resolved"
    resolution_bonus = 0.2 if is_resolved else 0.01  # never 0.0

    raw_score = (sentiment_score_component * sla_bonus) + (0.2 if is_resolved else 0.01)
    final_score = _clamp_score(raw_score)

    if is_resolved:
        feedback += " Ticket resolved, adding bonus."

    return Reward(
        score=final_score,
        partial_scores=_clamp_partial({
            "sentiment_component": sentiment_score_component,
            "sla_bonus": sla_bonus,
            "resolution_bonus": resolution_bonus,
        }),
        feedback=feedback,
        cumulative_score=_clamp_score(cumulative_score + final_score),
    )


def grade_queue_optimization(
    action: Dict[str, Any],
    initial_tickets: List[Ticket],
    final_tickets: List[Ticket],
    cumulative_score: float,
    done: bool,
    max_steps: int
) -> Reward:
    cumulative_score = _clamp_score(cumulative_score)

    if action.get("action_type") != "resolve":
        error_score = _clamp_score(0.01)
        return Reward(
            score=error_score,
            partial_scores=_clamp_partial({"no_resolve_action": 0.01}),
            feedback="No resolve action taken.",
            cumulative_score=_clamp_score(cumulative_score + error_score),
        )

    ticket_id_to_resolve = action.get("parameters", {}).get("ticket_id")
    if not ticket_id_to_resolve:
        error_score = _clamp_score(0.01)
        return Reward(
            score=error_score,
            partial_scores=_clamp_partial({"missing_ticket_id": 0.01}),
            feedback="Resolve action taken, but no ticket_id specified.",
            cumulative_score=_clamp_score(cumulative_score + error_score),
        )

    ticket_map = {t.ticket_id: t for t in initial_tickets}
    ticket = ticket_map.get(ticket_id_to_resolve)
    if ticket is None:
        error_score = _clamp_score(0.01)
        return Reward(
            score=error_score,
            partial_scores=_clamp_partial({"ticket_not_found": 0.01}),
            feedback=f"Ticket {ticket_id_to_resolve} not found.",
            cumulative_score=_clamp_score(cumulative_score + error_score),
        )

    raw_value = getattr(ticket, "value", 0.01)
    step_reward = _clamp_score(raw_value)
    feedback = f"Resolved ticket {ticket_id_to_resolve} with value {raw_value:.2f}."

    return Reward(
        score=step_reward,
        partial_scores=_clamp_partial({"resolved_value": raw_value}),
        feedback=feedback,
        cumulative_score=_clamp_score(cumulative_score + step_reward),
    )


GRADER_FUNCTIONS = {
    "sla_triage": grade_sla_triage,
    "sentiment_recovery": grade_sentiment_recovery,
    "queue_optimization": grade_queue_optimization,
}
