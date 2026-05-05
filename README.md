# CausalOS Discovery Worker

The **CausalOS Discovery Worker** is an asynchronous Python microservice that performs continuous structural causal discovery over agent execution ledgers.

Unlike standard LLM pattern matching or simple statistical correlation, this worker uses the **DoWhy** library to estimate the *true causal effect* of agent actions on system success or failure, adjusting for confounding variables.

## Architecture

This worker operates out-of-band from the real-time CausalOS Cloud Runtime to maintain the runtime's sub-15ms latency guarantees.

1. **Extraction**: Pulls episodic memory and execution data from the `causal_ledger` table.
2. **Modeling**: Constructs a structural causal model (DAG) linking:
   `action_type -> success_outcome`
3. **Estimation**: Runs a backdoor linear regression to determine the true causal weight of specific action types.
4. **Persistence**: Injects abstract rules back into the PostgreSQL `causal_nodes` and `causal_edges` tables using the special `GLOBAL_DISCOVERY` session ID.
5. **Inference**: The Rust `cloud-runtime` (`simulate_forward`) automatically detects these global weights and uses them to override naive empirical correlations with rigorous causal math.

## Setup

### Requirements
- Python 3.11+
- PostgreSQL (Supabase compatible)

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables
You must provide a valid PostgreSQL connection string. The worker supports standard connection URIs.
```bash
export DATABASE_URL="postgresql://user:password@host:port/dbname"
```

## Usage

Run the worker as a continuous background process:
```bash
python main.py
```

By default, the worker runs the batch discovery job once every hour.

## Why DoWhy?
Standard correlation fails when confounders are present (e.g., an action is often executed in a failing system state, making it *look* like the cause of the failure). By formalizing the causal graph, DoWhy allows CausalOS to mathematically isolate the true effect of the action itself.
