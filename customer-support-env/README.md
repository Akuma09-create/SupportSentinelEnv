# SupportSentinelEnv: An OpenEnv for AI Customer Support

Welcome to **SupportSentinelEnv**, an OpenEnv-compliant environment designed for the Meta + PyTorch Hackathon. This environment simulates the complex, real-world challenge of managing a customer support queue where an AI agent must balance two critical objectives: improving customer happiness (sentiment) and meeting Service Level Agreement (SLA) deadlines.

## Project Description and Motivation

In modern customer support, success isn't just about closing tickets. It's about resolving issues efficiently while ensuring customers feel heard and valued. A support interaction that takes too long can frustrate a customer, even if their problem is eventually solved. Conversely, a quick but unhelpful or cold response can be just as damaging.

This environment captures this dual objective by modeling:
1.  **Customer Sentiment**: A float score from -1.0 (furious) to +1.0 (delighted).
2.  **SLA Timers**: A step-based countdown for each ticket. Breaching an SLA incurs a significant penalty.

The agent's goal is to navigate these competing pressures, making strategic decisions to optimize for a combined score that reflects both sentiment and efficiency. This provides a rich testbed for developing sophisticated RL agents, planners, and multi-step reasoning models.

## File Structure

```
customer-support-env/
├── app.py              (FastAPI server with /reset /step /state endpoints)
├── environment.py      (Core SupportSentinelEnv class)
├── models.py           (Pydantic models: Action, Observation, Reward, EnvState)
├── tasks.py            (3 task definitions with ticket data)
├── graders.py          (Deterministic scoring logic)
├── inference.py        (Baseline script using OpenAI client)
├── openenv.yaml        (OpenEnv metadata)
├── Dockerfile          (HuggingFace Spaces compatible, port 7860)
├── requirements.txt
└── README.md
```

## How the SLA + Sentiment Dual Objective Works

-   **Sentiment**: Every action the agent takes can modify a ticket's sentiment score. The changes are deterministic and defined in the `SENTIMENT_CHANGE_TABLE`. For example, an empathetic response yields a higher sentiment boost than a formal one.
-   **SLA**: At every step of the simulation (for tasks 2 and 3), the `sla_steps_remaining` counter for ALL active tickets decreases by one. If it hits zero before the ticket is resolved, the ticket is flagged as `sla_breached`, and the agent is penalized.

The challenge lies in prioritizing. Should the agent focus on the angriest customer, or the one whose SLA is about to expire? This trade-off is the core of the environment.

## Action Space

The agent can perform one of the following actions at each step.

| Action | Parameters | Description |
| :--- | :--- | :--- |
| `prioritize` | `{"ticket_ids": ["t1", ...]` | (Task 1 Only) Submits a complete ordering of tickets. |
| `respond` | `{"ticket_id": "...", "tone": "..."}` | Responds to a ticket with a specific tone to influence sentiment. |
| `escalate` | `{"ticket_id": "...", "department": "..."}` | Escalates a ticket. Correct escalations boost sentiment, incorrect ones hurt it. |
| `compensate` | `{"ticket_id": "...", "type": "..."}` | Offers compensation, providing a significant sentiment boost. |
| `resolve` | `{"ticket_id": "...", "note": "..."}` | Closes the ticket. The final sentiment affects the reward. |
| `defer` | `{"ticket_id": "..."}` | Takes no action on the ticket, resulting in a small sentiment penalty. |

## Observation Space

At each step, the agent receives a detailed observation of the environment.

| Key | Type | Description |
| :--- | :--- | :--- |
| `tickets` | `List[Ticket]` | The full state of all tickets, including sentiment, SLA, history, etc. |
| `task_id` | `str` | The ID of the current task (e.g., "queue_optimization"). |
| `task_description` | `str` | A human-readable description of the task's goal. |
| `step_number` | `int` | The current step in the episode. |
| `max_steps` | `int` | The maximum number of steps allowed for this task. |
| `available_actions`| `List[str]` | The list of action types available for the current task. |
| `current_score` | `float` | The cumulative score achieved so far in the episode. |

## Task Descriptions

The environment includes three tasks of increasing difficulty.

### Task 1: SLA Triage (Easy)
-   **Goal**: Given 5 tickets with varying priorities and SLA deadlines, create a one-shot prioritization plan.
-   **Difficulty**: Easy, as it's a single-step planning problem. The agent only needs to correctly order the tickets based on urgency and SLA.
-   **Grader**: `score = % of tickets resolved within SLA` based on the provided order, assuming each ticket takes one step.

### Task 2: Sentiment Recovery (Medium)
-   **Goal**: Interact with a single, very angry customer (`sentiment = -0.7`) and raise their sentiment above `+0.3` before their 6-step SLA expires.
-   **Difficulty**: Medium. This is a sequential decision-making task focusing purely on sentiment management under a time constraint.
-   **Grader**: `(final_sentiment + 1) / 2 * sla_bonus`, where `sla_bonus` is proportional to the steps remaining.

### Task 3: Queue Optimization (Hard)
-   **Goal**: Manage a live queue of 6 simultaneous tickets. At each step, the agent must select one ticket and one action.
-   **Difficulty**: Hard. This is a complex resource allocation and scheduling problem. The agent must constantly re-evaluate priorities across the entire queue, balancing multiple sentiment scores and SLA countdowns that tick down together.
-   **Grader**: A weighted average: `(avg_sentiment_improvement * 0.4) + (sla_compliance_rate * 0.4) + (resolution_rate * 0.2)`.

## Setup and Installation

### Local Setup
1.  **Clone the repository** (or create the files as specified).
2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the FastAPI server**:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 7860
    ```
    The API documentation will be available at `http://localhost:7860/docs`.

### Docker Setup
1.  **Build the Docker image**:
    ```bash
    docker build -t supportsentinel .
    ```
2.  **Run the Docker container**:
    ```bash
    docker run -p 7860:7860 supportsentinel
    ```
    The server will be accessible at `http://localhost:7860`.

## API Reference

-   `GET /health`: Health check.
-   `GET /tasks`: Lists all available tasks.
-   `POST /reset`: Creates a new environment session.
    -   **Body**: `{ "task_id": "...", "seed": 42 }`
    -   **Returns**: The initial `Observation`. The `X-Session-Id` is returned in the response headers.
-   `POST /step?session_id=...`: Executes an action in the environment.
    -   **Body**: An `Action` object (e.g., `{"action_type": "respond", "parameters": {...}}`).
    -   **Returns**: A `StepResponse` containing the new observation, reward, done flag, and info.
-   `GET /state?session_id=...`: Retrieves the full state of an environment session.

## Baseline Scores

The `inference.py` script provides a baseline for agent performance using a standard LLM. It runs two episodes for each task (with seeds 42 and 43) and reports the average scores.

| Task | Difficulty | Baseline Average Score (Example) |
| :--- | :--- | :--- |
| `sla_triage` | Easy | *Run `inference.py` to generate* |
| `sentiment_recovery` | Medium | *Run `inference.py` to generate* |
| `queue_optimization` | Hard | *Run `inference.py` to generate* |
| **Overall Average** | | **Run `inference.py` to generate** |

To run the baseline test:
```bash
# Ensure the server is running in another terminal
python inference.py
```
You may need to set environment variables for the model and API key if you are not using the defaults.
```bash
export ENV_BASE_URL="http://your-deployed-url"
export MODEL_NAME="your-model-of-choice"
python inference.py
```
