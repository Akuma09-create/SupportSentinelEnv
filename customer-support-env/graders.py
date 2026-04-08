"""
Deterministic grading logic for SupportSentinelEnv tasks.
All returned scores are strictly within (0, 1).
"""
from typing import List, Dict, Any

try:
    from .models import Ticket, Reward
except (ImportError, ValueError):
    from models import Ticket, Reward


# 🔒 FINAL SAFE SCORE FUNCTION
def _safe_score(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.01

    if value <= 0:
        return 0.01
    if value >= 1:
        return 0.99

    return round(value, 4)


# ------------------- SLA TRIAGE -------------------

def grade_sla_triage(
    action: Dict[str, Any],
    initial_tickets: List[Ticket],
    final_tickets: List[Ticket],
    cumulative_score: float
) -> Reward:

    cumulative_score = _safe_score(cumulative_score)
    feedback_lines = []

    try:
        prioritized_ids = action["parameters"]["ticket_ids"]
        if not isinstance(prioritized_ids, list) or len(prioritized_ids) != len(initial_tickets):
            raise ValueError("Invalid ticket_ids format")
    except Exception as e:
        error_score = _safe_score(0.01)
        return Reward(
            score=error_score,
            partial_scores={"validation_error": error_score},
            feedback=f"Invalid action: {e}",
            cumulative_score=_safe_score(cumulative_score + error_score),
        )

    ticket_map = {t.ticket_id: t for t in initial_tickets}

    if len(set(prioritized_ids)) != len(initial_tickets) or set(prioritized_ids) != set(ticket_map.keys()):
        error_score = _safe_score(0.01)
        return Reward(
            score=error_score,
            partial_scores={"validation_error": error_score},
            feedback="ticket_ids must be unique and complete",
            cumulative_score=_safe_score(cumulative_score + error_score),
        )

    steps = 0
    success = 0
    total = len(initial_tickets)

    for tid in prioritized_ids:
        steps += 1
        ticket = ticket_map[tid]

        if steps <= ticket.sla_steps_remaining:
            success += 1
            feedback_lines.append(f"SUCCESS: {tid}")
        else:
            feedback_lines.append(f"FAIL: {tid}")

    raw_score = success / total if total > 0 else 0.01
    score = _safe_score(raw_score)

    return Reward(
        score=score,
        partial_scores={"sla_compliance": score},
        feedback="\n".join(feedback_lines),
        cumulative_score=_safe_score(cumulative_score + score),
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

    cumulative_score = _safe_score(cumulative_score)

    initial = initial_tickets[0]
    final = final_tickets[0]

    if not done:
        change = final.sentiment_score - initial.sentiment_score
        raw = 0.5 + (change * 0.5)
        score = _safe_score(raw)

        return Reward(
            score=score,
            partial_scores={"sentiment_change": score},
            feedback=f"Change: {change:.2f}",
            cumulative_score=_safe_score(cumulative_score + score),
        )

    sentiment = _safe_score((final.sentiment_score + 1) / 2.0)

    if final.sla_breached:
        sla_bonus = _safe_score(0.5)
    else:
        raw_sla = (
            final.sla_steps_remaining / final.sla_total_steps
            if final.sla_total_steps > 0 else 0.01
        )
        sla_bonus = _safe_score(raw_sla)

    resolved = final.status == "resolved"
    resolution_bonus = _safe_score(0.2 if resolved else 0.01)

    raw_score = (sentiment * sla_bonus) + resolution_bonus
    final_score = _safe_score(raw_score)

    return Reward(
        score=final_score,
        partial_scores={
            "sentiment_component": sentiment,
            "sla_bonus": sla_bonus,
            "resolution_bonus": resolution_bonus,
        },
        feedback=f"Final sentiment: {final.sentiment_score:.2f}",
        cumulative_score=_safe_score(cumulative_score + final_score),
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

    cumulative_score = _safe_score(cumulative_score)

    if action.get("action_type") != "resolve":
        err = _safe_score(0.01)
        return Reward(
            score=err,
            partial_scores={"no_resolve_action": err},
            feedback="No resolve action taken.",
            cumulative_score=_safe_score(cumulative_score + err),
        )

    tid = action.get("parameters", {}).get("ticket_id")
    if not tid:
        err = _safe_score(0.01)
        return Reward(
            score=err,
            partial_scores={"missing_ticket_id": err},
            feedback="Missing ticket_id.",
            cumulative_score=_safe_score(cumulative_score + err),
        )

    tickets = {t.ticket_id: t for t in initial_tickets}
    ticket = tickets.get(tid)

    if ticket is None:
        err = _safe_score(0.01)
        return Reward(
            score=err,
            partial_scores={"ticket_not_found": err},
            feedback=f"Ticket {tid} not found.",
            cumulative_score=_safe_score(cumulative_score + err),
        )

    raw = getattr(ticket, "value", 0.01)
    score = _safe_score(raw)

    return Reward(
        score=score,
        partial_scores={"resolved_value": score},
        feedback=f"Resolved {tid}",
        cumulative_score=_safe_score(cumulative_score + score),
    )


# ------------------- REGISTRY -------------------

GRADER_FUNCTIONS = {
    "sla_triage": grade_sla_triage,
    "sentiment_recovery": grade_sentiment_recovery,
    "queue_optimization": grade_queue_optimization,
}
