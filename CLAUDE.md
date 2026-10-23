# MultitaskOCTA — notes for Claude Code sessions

## Session wrap-up (required)
At the end of every session that changes code, results or docs:
1. Update `docs/STATUS.md`: current state, environment, results, skills demonstrated, open issues, next steps.
2. Facts only. Every number needs a source file or commit. Do not write anything the code, logs or committed results do not show.
3. For each skill used, add a "Skills demonstrated" row with the file or commit as evidence.
4. Keep the header line under "Last updated" in `docs/STATUS.md`: `Branch: fix/thesis-consistency · Commit: <short hash>`, where the hash is `git rev-parse --short HEAD` taken just before committing (the commit the status describes; the commit that carries the file sits on top of it).
5. Every number that may go on a CV, LinkedIn or the README must have a row in `docs/CLAIMS.md` (claim, number, exact metric name, source, status, safe wording, wording to avoid). Accuracy and balanced accuracy are never compared across the thesis and a re-run. Mark anything untested as hypothesis. Update the table in the same commit as a changed result.
6. Commit `docs/STATUS.md` and `docs/CLAIMS.md` together with the work.
7. Push to `origin` on the current branch (`fix/thesis-consistency`, the repository's default branch; the Main Professional project syncs `README.md`, `CLAUDE.md` and `docs/` from it). Do not merge into or push to `master` unless Kourosh explicitly asks for it in that session: he decided that `master` (the thesis-time code) stays untouched. Keep those paths stable and say so before renaming `docs/`.
8. Do not change README claims beyond what `docs/STATUS.md` and `docs/CLAIMS.md` support.
9. CV and LinkedIn text is written in the Main Professional project, not here: no local CV drafts. End the wrap-up with one line: `Pushed <branch>@<hash>; click Sync in Main Professional.`

## Repository rules that already hold
- Run `SKIP_SLOW_TESTS=1 python -m unittest discover -s tests` before a commit; the golden regression tests (`GOLDEN_TOL=1e-9 python -m unittest discover -s tests`) after any change of training code. Results must stay bit-identical unless a change of behaviour is intended and documented.
- Stage explicit paths only (never `git add -A`); several local, untracked files (tracker, notes) are not part of the repository.
- Do not publish OCTA-500 scan images (the dataset terms say nothing about it); figures use masks and plots only.
