"""
Script to replace graders.py with explicit intermediate value clamping.
This is the ULTIMATE fix for boundary value violations.
"""

new_content = '''"""
Deterministic grading logic for SupportSentinelEnv tasks.
All returned scores are strictly within (0, 1).
"""
from typing import List, Dict, Any

try:
    from .models import Ticket, Reward
except (ImportError, ValueError):
    from models import Ticket, Reward


def _clamp_score(value: float, min_val: float = 0.001, max_val: float = 0.999) -> float:
    """Clamp a numeric score strictly within (0, 1)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return min_val
    return max(min_val, min(max_val, value))


# ------------------- SLA TRIAGE -------------------

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
    except Exception as e:
        error_score = _clamp_score(0.01)
        return Reward(
            score=error_score,
            partial_scores={"validation_error": error_score},
            feedback=f"Invalid action format: {e}",
            cumulative_score=_clamp_score(cumulative_score + error_score),
        )

    ticket_map = {t.ticket_id: t for t in initial_tickets}

    if len(set(prioritized_ids)) != len(initial_tickets) or set(prioritized_ids) != set(ticket_map.keys()):
        error_score = _clamp_score(0.01)
        return Reward(
            score=error_score,
            partial_scores={"validation_error": error_score},
            feedback="ticket_ids must contain all unique ticket IDs exactly once.",
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
            feedback_lines.append(f"SUCCESS: {ticket_id} resolved at step {steps_elapsed}.")
        else:
            feedback_lines.append(f"FAILURE: {ticket_id} breached at step {steps_elapsed}.")

    raw_score = resolved_within_sla / total_tickets if total_tickets > 0 else 0.001
    score = _clamp_score(raw_score)

    feedback = "SLA Triage Result:\\n" + "\\n".join(feedback_lines)

    return Reward(
        score=score,
        partial_scores={"sla_compliance": score},  # SAFE (already clamped)
        feedback=feedback,
        cumulative_score=_clamp_score(cumulative_score + score),
    )


# ------------------- SENTIMENT RECOVERY -------------------

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

    # Step-wise reward
    if not done:
        sentiment_change = final_ticket.sentiment_score - initial_ticket.sentiment_score
        raw_step_score = 0.5 + (sentiment_change * 0.5)
        step_score = _clamp_score(raw_step_score)

        return Reward(
            score=step_score,
            partial_scores={"sentiment_change": step_score},
            feedback=f"Sentiment change: {sentiment_change:.2f}",
            cumulative_score=_clamp_score(cumulative_score + step_score),
        )

    # Final reward - CRITICAL: clamp intermediates BEFORE Reward creation
    final_sentiment = final_ticket.sentiment_score
    sentiment_component = _clamp_score((final_sentiment + 1) / 2.0)

    if final_ticket.sla_breached:
        sla_bonus = _clamp_score(0.5)
    else:
        raw_sla = (
            final_ticket.sla_steps_remaining / final_ticket.sla_total_steps
            if final_ticket.sla_total_steps > 0 else 0.001
        )
        sla_bonus = _clamp_score(raw_sla)

    is_resolved = final_ticket.status == "resolved"
    resolution_bonus = _clamp_score(0.2 if is_resolved else 0.01)

    raw_score = (sentiment_component * sla_bonus) + resolution_bonus
    final_score = _clamp_score(raw_score)

    feedback = f"Final sentiment: {final_sentiment:.2f}"
    if is_resolved:
        feedback += " | Ticket resolved"

    return Reward(
        score=final_score,
        partial_scores={
            "sentiment_component": sentiment_component,
            "sla_bonus": sla_bonus,
            "resolution_bonus": resolution_bonus,
        },
        feedback=feedback,
        cumulative_score=_clamp_score(cumulative_score + final_score),
    )


# ------------------- QUEUE OPTIMIZATION -------------------

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
            partial_scores={"no_resolve_action": error_score},
            feedback="No resolve action taken.",
            cumulative_score=_clamp_score(cumulative_score + error_score),
        )

    ticket_id = action.get("parameters", {}).get("ticket_id")
    if not ticket_id:
        error_score = _clamp_score(0.01)
        return Reward(
            score=error_score,
            partial_scores={"missing_ticket_id": error_score},
            feedback="Missing ticket_id.",
            cumulative_score=_clamp_score(cumulative_score + error_score),
        )

    ticket_map = {t.ticket_id: t for t in initial_tickets}
    ticket = ticket_map.get(ticket_id)

    if ticket is None:
        error_score = _clamp_score(0.01)
        return Reward(
            score=error_score,
            partial_scores={"ticket_not_found": error_score},
            feedback=f"Ticket {ticket_id} not found.",
            cumulative_score=_clamp_score(cumulative_score + error_score),
        )

    raw_value = getattr(ticket, "value", 0.01)
    step_reward = _clamp_score(raw_value)

    return Reward(
        score=step_reward,
        partial_scores={"resolved_value": step_reward},
        feedback=f"Resolved {ticket_id}",
        cumulative_score=_clamp_score(cumulative_score + step_reward),
    )


# ------------------- REGISTRY -------------------

GRADER_FUNCTIONS = {
    "sla_triage": grade_sla_triage,
    "sentiment_recovery": grade_sentiment_recovery,
    "queue_optimization": grade_queue_optimization,
}
'''

# Write the fixed content
with open("customer-support-env/graders.py", "w") as f:
    f.write(new_content)

print("✅ graders.py replaced with ULTIMATE fix!")
print("✅ All intermediate values explicitly clamped BEFORE Reward creation")
print("✅ sentiment_component clamped")
print("✅ sla_bonus clamped")
print("✅ resolution_bonus clamped")
print("✅ ALL partial_scores contain pre-clamped values")
print("✅ Bounds: [0.001, 0.999] for maximum safety")
