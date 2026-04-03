"""
Ticket data and task definitions for the SupportSentinelEnv.
"""
from typing import Dict, List
from .models import Ticket

# --- Task 1: SLA Triage ---
SLA_TRIAGE_TICKETS: List[Dict] = [
    {
        "ticket_id": "t1",
        "subject": "Urgent: Billing Discrepancy",
        "body": "My latest invoice is incorrect. I was overcharged and need this fixed immediately as it affects my budget.",
        "customer_name": "Alpha Corp",
        "customer_tier": "enterprise",
        "category": "billing",
        "sentiment_score": -0.5,
        "sla_steps_remaining": 2,
        "sla_total_steps": 2,
    },
    {
        "ticket_id": "t2",
        "subject": "Critical Outage: API Not Responding",
        "body": "Our service is down because your API is returning 500 errors. This is a major incident for us.",
        "customer_name": "Beta Solutions",
        "customer_tier": "pro",
        "category": "technical",
        "sentiment_score": -0.8,
        "sla_steps_remaining": 4,
        "sla_total_steps": 4,
    },
    {
        "ticket_id": "t3",
        "subject": "Can't reset my password",
        "body": "I'm trying to reset my password but the link seems to be broken. Can you help?",
        "customer_name": "Charlie User",
        "customer_tier": "free",
        "category": "account",
        "sentiment_score": -0.2,
        "sla_steps_remaining": 7,
        "sla_total_steps": 7,
    },
    {
        "ticket_id": "t4",
        "subject": "SEVERE: Data Loss Incident",
        "body": "A portion of our user data from last night appears to be missing. This is a top priority security and data integrity issue.",
        "customer_name": "Gamma Industries",
        "customer_tier": "enterprise",
        "category": "technical",
        "sentiment_score": -0.9,
        "sla_steps_remaining": 3,
        "sla_total_steps": 3,
    },
    {
        "ticket_id": "t5",
        "subject": "Feature Request: Dark Mode",
        "body": "I love the platform, but it would be great if you could add a dark mode. My eyes would thank you!",
        "customer_name": "Delta LLC",
        "customer_tier": "pro",
        "category": "account",
        "sentiment_score": 0.1,
        "sla_steps_remaining": 5,
        "sla_total_steps": 5,
    },
]

# --- Task 2: Sentiment Recovery ---
SENTIMENT_RECOVERY_TICKET: List[Dict] = [
    {
        "ticket_id": "t_angry",
        "subject": "Outraged: You double-charged my account!",
        "body": "I just checked my statement and you have charged me twice for this month's subscription. This is completely unacceptable. I demand an immediate refund and an explanation.",
        "customer_name": "Angry Enterprise",
        "customer_tier": "enterprise",
        "category": "billing",
        "sentiment_score": -0.7,
        "sla_steps_remaining": 6,
        "sla_total_steps": 6,
    }
]

# --- Task 3: Queue Optimization ---
QUEUE_OPTIMIZATION_TICKETS: List[Dict] = [
    {
        "ticket_id": "q1",
        "subject": "URGENT: Production Server Down",
        "body": "Our main production server is offline. We are losing business every minute. What is the status?",
        "customer_name": "Omega Corp",
        "customer_tier": "enterprise",
        "category": "technical",
        "sentiment_score": -0.8,
        "sla_steps_remaining": 3,
        "sla_total_steps": 3,
    },
    {
        "ticket_id": "q2",
        "subject": "Shipment delayed by 2 weeks",
        "body": "My package was supposed to arrive last week and the tracking hasn't updated. Where is it?",
        "customer_name": "Freebie Fred",
        "customer_tier": "free",
        "category": "shipping",
        "sentiment_score": -0.4,
        "sla_steps_remaining": 8,
        "sla_total_steps": 8,
    },
    {
        "ticket_id": "q3",
        "subject": "Question about my bill",
        "body": "I have a question about a line item on my recent invoice. Can someone from billing clarify it for me?",
        "customer_name": "Pro User Penny",
        "customer_tier": "pro",
        "category": "billing",
        "sentiment_score": -0.1,
        "sla_steps_remaining": 5,
        "sla_total_steps": 5,
    },
    {
        "ticket_id": "q4",
        "subject": "Login issues",
        "body": "I can't log into my account. It says 'invalid credentials' but I am sure I am using the right password.",
        "customer_name": "Standard Sam",
        "customer_tier": "free",
        "category": "account",
        "sentiment_score": -0.3,
        "sla_steps_remaining": 8,
        "sla_total_steps": 8,
    },
    {
        "ticket_id": "q5",
        "subject": "Positive feedback!",
        "body": "Just wanted to say your new feature is amazing! It's saved my team so much time. Keep up the great work.",
        "customer_name": "Happy Helen",
        "customer_tier": "pro",
        "category": "general",
        "sentiment_score": 0.8,
        "sla_steps_remaining": 5,
        "sla_total_steps": 5,
    },
    {
        "ticket_id": "q6",
        "subject": "Wrong item received",
        "body": "I ordered a blue widget but received a red one. How do I arrange for a return and replacement?",
        "customer_name": "Shipper Sheila",
        "customer_tier": "pro",
        "category": "shipping",
        "sentiment_score": -0.6,
        "sla_steps_remaining": 4,
        "sla_total_steps": 4,
    },
]


TASK_DEFINITIONS = {
    "sla_triage": {
        "description": "Prioritize a list of 5 tickets by urgency to maximize SLA compliance. Assumes each ticket takes 1 step to resolve.",
        "difficulty": "Easy",
        "max_steps": 1,
        "tickets": [Ticket(**t) for t in SLA_TRIAGE_TICKETS],
        "available_actions": ["prioritize"],
    },
    "sentiment_recovery": {
        "description": "Handle a single angry customer and raise their sentiment from -0.7 to > +0.3 before the SLA of 6 steps expires.",
        "difficulty": "Medium",
        "max_steps": 8,
        "tickets": [Ticket(**t) for t in SENTIMENT_RECOVERY_TICKET],
        "available_actions": ["respond", "escalate", "compensate", "resolve", "defer"],
    },
    "queue_optimization": {
        "description": "Manage a queue of 6 tickets simultaneously. Each step, choose one ticket and one action to optimize sentiment, SLA, and resolution rate.",
        "difficulty": "Hard",
        "max_steps": 15,
        "tickets": [Ticket(**t) for t in QUEUE_OPTIMIZATION_TICKETS],
        "available_actions": ["respond", "escalate", "compensate", "resolve", "defer"],
    },
}
