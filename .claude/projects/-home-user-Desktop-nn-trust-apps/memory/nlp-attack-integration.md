---
name: nlp-attack-integration
description: Backend and frontend refactoring to support NLP/Jailbreak tasks in TITANN
metadata:
  type: project
---

**Why:** NLP attacks (e.g., GCG, PAIR) from `nn_trust` required a different execution pathway from existing image-based classification attacks.
**How to apply:**
- `ExecutionConfig` (Pydantic model) now requires `task_type` (default "classification").
- The backend dispatches jobs based on `task_type`.
- For `nlp` tasks, `execution.py` uses `get_dataloader_nlp` (which loads JSON goals) and `evaluate_attack_nlp`, bypassing vision-centric transforms.
- NLP attacks are filtered by `Task.Language` in `info_router.py`.
- Frontend `frontend` automatically detects model type (`llm` vs `cv`) to propagate `task_type` ("nlp" or "classification") to the backend via `handlePostRequest` API calls.
