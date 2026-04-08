---
title: SupportFlow CRM
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
---
# SupportFlow CRM: Enterprise Logistics RL Environment

SupportFlow is a real-world task simulation designed for AI agents to perform complex customer support and logistics triage . Unlike toy problems, this environment requires multi-step reasoning across disconnected data points—Users, Orders, and Shipments—to resolve high-stakes customer issues like lost packages and automated refund processing .

## 🚀 Environment Overview & Motivation
In modern enterprise settings, support agents must navigate multiple APIs to solve a single ticket. SupportFlow replicates this complexity to test an agent's ability to:
1. **Gather Context**: Authenticate users and retrieve historical order data.
2. **Investigate**: Query logistics providers for real-time shipping updates.
3. **Execute Policy**: Perform financial actions based on specific "Ground Truth" conditions (e.g., only refunding "LOST" items).

This environment provides a robust testbed for Agentic AI in a safe, reproducible sandbox that mirrors actual professional workflows.

## 🛠️ Action & Observation Spaces

### Action Space (ActionRequest)
The environment exposes a set of Pydantic-typed tools for the agent:
* **`search_user`**: Retrieve customer profiles via email.
* **`get_order_details`**: Fetch line items and status for a specific Order ID.
* **`track_package`**: Query real-time shipping status (e.g., DELIVERED, IN_TRANSIT, LOST).
* **`issue_refund`**: Execute a financial transaction (requires 'LOST' shipment status).
* **`respond`**: Finalize the ticket with a message to the customer (Terminal Action).
### Observation Space (Observation)
After each action, the environment returns a typed observation:
* **`display_output`**: Textual feedback from the CRM system (e.g., "USER_FOUND").
* **`last_action_error`**: Raw error strings if a tool call fails (or `null` if successful).
* **`is_terminal`**: Boolean indicating if the support ticket is closed.
## 📋 Task Descriptions
SupportFlow includes three distinct tasks with increasing difficulty :
| Task ID | Name | Difficulty | Objective |
| :--- | :--- | :--- | :--- |
| `easy_tracking` | Retrieve Tracking | Easy | Find and report the tracking number for a user. |
| `medium_policy_check` | Refund Audit | Medium | Verify if an order meets the 'LOST' criteria for a refund. |
| `hard_full_resolution` | Lost Package | Hard | Identify a lost item, process a refund, and notify the user. |
Each task features a programmatic grader that assigns a deterministic score between **0.0 and 1.0** based on backend state changes.

## 💰 Reward Function
The environment implements a **Meaningful Reward Function** providing feedback throughout the trajectory :
* **Incremental Progress**: Rewards are granted for successful data retrieval (e.g., `+0.20` for search).
* **Root Cause Discovery**: Significant reward (`+0.30`) for identifying a 'LOST' shipment.
* **Goal Completion**: Final reward (`+0.50`) for a justified refund.
* **Penalties**: Negative rewards (`-0.20`) are applied for policy violations or infinite loops.

## 📈 Baseline Performance
Evaluation results using **Qwen2.5-72B-Instruct**:
* **Task**: `hard_full_resolution`
* **Success Rate**: 100%
* **Steps Taken**: 5
* **Baseline Score**: **1.000**

## Baseline Logs
```text
[START] task=hard_full_resolution env=support-flow-crm model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"command": "search_user", "params": {"email": "customer@example.com"}} reward=0.20 done=false error=null
[STEP] step=2 action={"command": "get_order_details", "params": {"order_id": "ord_101"}} reward=0.20 done=false error=null
[STEP] step=3 action={"command": "track_package", "params": {"order_id": "ord_101"}} reward=0.30 done=false error=null
[STEP] step=4 action={"command": "issue_refund", "params": {"order_id": "ord_101"}} reward=0.50 done=false error=null
[STEP] step=5 action={"command": "respond", "params": {"message": "We have processed a refund of $150.0 for your order ord_101. The refund will be credited to your original payment method. We apologize for the inconvenience and appreciate your understanding."}} reward=1.00 done=true error=null
[END] success=true steps=5 rewards=0.20,0.20,0.30,0.50,1.00
```

## 💻 Setup and Usage

### Prerequisites
* Python 3.10+
* Docker (for containerized execution) 

### Local Installation
```bash
# Clone the repository and enter the directory
pip install -r requirements.txt
python app.py
