# Log Investigator
[Detailed Design Doc](https://docs.google.com/document/d/1AqO-vcmYFUWDHxDJMe3qmtmxWTnqByCnfKXvzhINZTU/edit?usp=sharing)
## Overview

![log_inv](https://github.com/user-attachments/assets/9d0b25e1-3ab7-4311-b03a-862993f3205b)


This project trains an incident‑investigation agent using the ART framework.  
The agent receives a vague incident scenario and must investigate structured logs using two tools:

- `search_logs`
- `read_trace`

It must discover the technical **root cause** and return a structured **final JSON report** via the `return_final_answer` tool.

Training uses synthetic scenarios, a SQLite log database, an LLM judge for correctness scoring, and a reward function designed to encourage deep reasoning and penalize tool misuse.

---

## Repository Structure
- `train_serverless.py` — Main ART training loop  
- `rollout.py` — Multi‑turn investigation logic + reward calculation  
- `project_types.py` — Pydantic config models  
- `log_search_tools.py` — Tools (search_logs, read_trace) with scenario isolation  
- `build_logs.py` — Builds `logs.db` from scenario JSON files  
- `make_judge_model.py` — Creates/registers the judge model  
- `log_scenarios/train/` — Training scenarios + combined JSONL  

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set required environment variables
```bash
export WANDB_API_KEY="..."
# set after creating the judge model:
export JUDGE_MODEL="..."
```

### 3. Build the logs database
This loads all `scenario-*.json` files from `log_scenarios/train/` into a single SQLite DB.
```bash
python3 build_logs.py
```

### 4. Register the judge model
This creates the LLM-as-judge used to score rollouts.
```bash
python3 make_judge_model.py
export JUDGE_MODEL="your_judge_model_name"
```

### 5. Train the agent
```bash
python3 train_serverless.py
```

---

## Notes
- Re‑run `build_logs.py` whenever scenario JSON files change.  
- `search_logs` and `read_trace` enforce per‑scenario isolation via `scenario_id`.  
- Reward penalizes: malformed tool calls, unsupported guesses, and "unknown" answers.  
- Every scenario is guaranteed to contain a real root cause in the logs.  
- The agent can take up to 10 investigation turns per scenario.
