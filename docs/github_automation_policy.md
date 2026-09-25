# GITHUB AUTOMATION POLICY — AMAZON ML CHALLENGE

1. **BRANCHING**
   - Never perform experimental work directly on main.
   - Use: `main`, `feature/*`, `experiment/*`
   - Recommended naming: `experiment/aayush-blocking-<experiment>`, `experiment/ashank-model-<experiment>`, `feature/mudit-validation-<feature>`
   - Keep main stable and reproducible.

2. **BEFORE STARTING WORK**
   - Read `AI_CONTEXT.md`
   - Inspect git status, current branch, recent commits, relevant existing code.
   - Check whether another teammate has already implemented the requested functionality.
   - Do not duplicate existing work.

3. **AFTER IMPLEMENTATION**
   - Run relevant tests and validation.
   - Inspect git diff and newly created files.
   - Remove accidental/generated clutter.
   - Verify no secrets, datasets, or PDFs are present and `.gitignore` is working.

4. **COMMIT AUTOMATION**
   - Create a descriptive commit. Keep commits logically scoped.
   - Examples: `feat: implement TF-IDF candidate blocking`, `exp: benchmark character n-gram blocking`.

5. **PUSH AUTOMATION**
   - Push the current feature/experiment branch to GitHub.
   - Verify the remote branch exists and report the commit hash and branch name.

6. **PULL REQUESTS**
   - Create a Pull Request against main.
   - Include: objective, hypothesis, changes, validation results, metrics (F0.5, recall, runtime, memory), risks.

7. **MAIN BRANCH PROTECTION**
   - Do NOT merge a feature/experiment into main automatically merely because the code works.
   - Wait for Mudit's approval before merging. Verify tests, reproducibility, docs, and regressions.

8. **EXPERIMENT VERSIONING**
   - Every meaningful experiment must be traceable to: experiment ID, commit, branch, configuration, version, score.

9. **TEAM BRANCHES**
   - Mudit: integration/validation
   - Aayush: blocking
   - Ashank: matching/model
   - Do not modify another teammate's branch unless explicitly instructed.

10. **MERGING TEAMMATE WORK**
   - Mudit decides when to integrate. Inspect branch, diff, run validation, resolve conflicts carefully.

11. **GITHUB DOCUMENTATION**
   - Maintain `README.md`, `AI_CONTEXT.md`, experiment documentation, `methodology.md`.

12. **AI_CONTEXT.md SYNCHRONIZATION**
   - After every major completed task or experiment, update `AI_CONTEXT.md` with: what changed, experiment ID, branch, commit, results, current best result, next recommended experiment.

13. **CLEANUP**
   - Remove temporary scripts/files, move useful artifacts, keep notebooks organized.

14. **FAILURE HANDLING**
   - If tests/validation fail, conflicts occur, or unexpected changes appear: STOP before pushing/merging. Investigate and report.

15. **AUTONOMY PRINCIPLE**
   - Authorized to autonomously perform routine Git operations (status, diff, add, commit, branch, push, PR creation).
   - Mudit is the final integration authority. Do not automatically merge major work into main.
