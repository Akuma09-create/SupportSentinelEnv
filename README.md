---
title: SupportSentinelEnv
emoji: 🏢
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
---
# 🚀 SupportSentinelEnv: An AI Customer Support Environment

**`SupportSentinelEnv`** is a real-world simulation of a customer support ticket system, built for the **OpenEnv Hackathon** by Meta, Hugging Face, and Scaler. This environment allows AI agents to learn and operate using a standardized API, tackling challenges that mirror real-world customer service operations.

This project demonstrates how AI can be trained to automate support workflows, improve efficiency, and enhance user experience in a controlled, reproducible setting.

## 🎯 Project Objective

The goal of this environment is to provide a platform for training and evaluating AI agents on their ability to:

*   **Prioritize** customer support tickets based on urgency and SLA.
*   **Resolve** issues efficiently to optimize queue throughput.
*   **Improve** customer satisfaction by analyzing sentiment and choosing appropriate actions.
*   **Operate** under constraints like time limits and available actions.

## ✨ Key Features

*   **Real-World Simulation**: Models a dynamic customer support queue with tickets of varying sentiment, priority, and value.
*   **OpenEnv Compliant**: Follows the standard `reset()` and `step(action)` API, making it compatible with modern agent-based frameworks.
*   **Structured Data**: Utilizes Pydantic models for clear, typed observation and action spaces.
*   **Deterministic Grading**: Provides a reliable scoring system (0.0–1.0) to measure agent performance objectively.
*   **Three Core Tasks**: Offers distinct challenges to test different agent capabilities.

---

## ⚙️ Getting Started

Follow these steps to set up and run the environment locally.

### 1. Prerequisites

*   Python 3.8+
*   `pip` for package management

### 2. Installation

First, set up a virtual environment to keep dependencies isolated:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate
```

Next, install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Running the Environment Server

The environment is served via a FastAPI application. To start the server, run the following command in your terminal:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

The server will now be running and ready to accept requests at `http://localhost:7860`.

### 4. Running the Baseline Agent

To see the environment in action, you can run the baseline inference script. This script will interact with the running server, complete all three tasks, and print the scores.

Open a **new terminal** (while the server is still running in the first one) and run:

```bash
python -m customer-support-env.inference
```

You will see the agent run through each task and output the final scores and rewards for each episode.

---

## 🤖 Environment Tasks

`SupportSentinelEnv` includes three distinct tasks designed to test an agent's decision-making skills in different scenarios.

### 1. SLA Triage

*   **Objective**: Prioritize a list of tickets to resolve as many as possible before their Service Level Agreement (SLA) deadlines expire.
*   **Challenge**: The agent must correctly order the entire list of tickets in a single action. The score is based on the percentage of tickets that would be resolved within their SLA given that order.

### 2. Sentiment Recovery

*   **Objective**: Improve a single unhappy customer's sentiment by choosing the right sequence of actions.
*   **Challenge**: The agent interacts with a ticket that has a negative sentiment score. It must choose actions (e.g., responding with an "empathetic" tone) to increase the customer's sentiment score over several steps.

### 3. Queue Optimization

*   **Objective**: Resolve a queue of tickets in the most valuable order.
*   **Challenge**: Each ticket has a different "value". The agent is rewarded for each ticket it successfully resolves, with the reward being equal to the ticket's value. The goal is to maximize the total value of resolved tickets within the allowed steps.

