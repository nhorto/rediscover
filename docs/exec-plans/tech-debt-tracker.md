# Tech Debt Tracker

> Known technical debt items. Log debt here, don't let it hide.

| Item | Severity | Domain | Description | Logged |
|------|----------|--------|-------------|--------|
| Git state collision | Medium | app/loop | Running the loop while making development commits causes git commit failures. The loop should run on a separate branch or use worktrees. | 2026-03-17 |
| MuonAdamW shape error | Medium | experiments | Optimizer crashes with IndexError on 1D parameters when the attention code adds parameters with <2 dimensions. Need to ensure new params go in the right optimizer group. | 2026-03-17 |
