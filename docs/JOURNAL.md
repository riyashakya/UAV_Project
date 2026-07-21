# Work journal

A structured, chronological record of work. **Every task is journaled here _before_ it is
started**, for traceability and easy resumption. Newest entries at the top.

Entry format: **Request** · **Summary** · **Root cause / motivation** · **Solution** ·
**Why this solution** · **Files changed** · **Status**.

Related: [`ROADMAP.md`](ROADMAP.md) (phase status), [`BUILD_LOG.md`](BUILD_LOG.md) (earlier
detailed technical log), [`adr/`](adr/) (architecture decisions).

---

## 2026-07-21 — Push repository to GitHub

- **Request:** push all work so far to `https://github.com/riyashakya/UAV_Project.git`.
- **Summary:** add the remote and push `main` (code + docs only; datasets/weights are gitignored).
- **Root cause / motivation:** off-machine backup and supervisor visibility; the repo had no remote.
- **Solution:** `git remote add origin <url>` then `git push -u origin main`.
- **Why this solution:** standard first push; nothing large or sensitive travels (data/ and *.pt
  are ignored). Credentials must come from the user's own git/GitHub auth — never entered here.
- **Files changed:** none (git metadata only; adds `origin` remote).
- **Status:** ✅ done — pushed to `origin/main` (13 commits). Future work: `git push` keeps it synced.

## 2026-07-21 — Project-tracking docs, progress report, and strategic review

- **Request:** (1) explain what segmentation is for; (2) create a rich change/decision log;
  (3) push everything to git; (4) a markdown file tracking finished plans; (5) journal work
  before starting it, going forward; (6) how good are the models; (7) my contribution vs
  existing work + how to improve; (8) the research question; (9) a dissertation progress
  report; (10-12) whether to focus/deepen features vs breadth, which ones, and whether the
  project is heading toward "mediocre".
- **Summary:** Established the journaling discipline (this file), added a phase-status tracker
  and an interim progress report, and gave an honest depth-vs-breadth assessment. Ensured all
  work is committed; documented the git-remote gap for pushing.
- **Root cause / motivation:** The project has produced solid setup + perception results but no
  novel *contribution* yet, and the student is (rightly) asking whether breadth across 7
  objectives risks a mediocre outcome. Also needs auditable history for the write-up.
- **Solution:** Created `docs/JOURNAL.md` (this), `docs/ROADMAP.md`, `docs/PROGRESS_REPORT.md`;
  saved the journal-before-work rule to memory; committed everything. Push to a GitHub remote
  is blocked (no remote, `gh` not installed) — handed off with exact commands.
- **Why this solution:** JOURNAL + ROADMAP separate "what happened / why" from "what's done vs
  pending"; the progress report consolidates research question, results, and the honest
  contribution analysis in one citable place. Kept `BUILD_LOG.md` for detailed history rather
  than deleting it.
- **Files changed:** `docs/JOURNAL.md` (new), `docs/ROADMAP.md` (new),
  `docs/PROGRESS_REPORT.md` (new), memory `journaling-workflow.md` (new).
- **Status:** ✅ done (docs); ⏳ push pending a remote (see ROADMAP / chat).

## Earlier work (pre-journal)

Phases 0–2 (scaffold, dataset unification, perception training) predate this journal and are
recorded in [`BUILD_LOG.md`](BUILD_LOG.md). Headline: both YOLO11 models trained —
Model A (detect) mAP@50 0.674, Model B (segment) mask mAP@50 0.410.
