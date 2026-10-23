# MultitaskOCTA — notes for Claude Code sessions

## Session wrap-up (required)
At the end of every session that changes code, results or docs:
1. Update `docs/STATUS.md`: current state, environment, results, skills demonstrated, open issues, next steps.
2. Facts only. Every number needs a source file or commit. Do not write anything the code, logs or committed results do not show.
3. For each skill used, add a "Skills demonstrated" row with the file or commit as evidence.
4. Commit `docs/STATUS.md` together with the work.
5. Push to `origin` on the current branch (`fix/thesis-consistency`, the repository's default branch). Do not merge into or push to `master` unless Kourosh explicitly asks for it in that session: he decided that `master` (the thesis-time code) stays untouched. Work that stays on a local branch is not visible on GitHub.
6. Do not change README claims beyond what `docs/STATUS.md` supports.

## Repository rules that already hold
- Run `SKIP_SLOW_TESTS=1 python -m unittest discover -s tests` before a commit; the golden regression tests (`GOLDEN_TOL=1e-9 python -m unittest discover -s tests`) after any change of training code. Results must stay bit-identical unless a change of behaviour is intended and documented.
- Stage explicit paths only (never `git add -A`); several local, untracked files (tracker, notes) are not part of the repository.
- Do not publish OCTA-500 scan images (the dataset terms say nothing about it); figures use masks and plots only.
