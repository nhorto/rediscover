---
name: Use cloud for training
description: User wants all training done on Modal cloud GPU, not local compute
type: feedback
---

Use Modal cloud GPU (--runner modal) for all experiment training runs, not local compute.

**Why:** User doesn't want to use their local computer's compute for training experiments.

**How to apply:** Always use `--runner modal` flag when running experiments. Modal app is deployed at modal.com/apps/nickbaxter5757/main/deployed/rediscover.
