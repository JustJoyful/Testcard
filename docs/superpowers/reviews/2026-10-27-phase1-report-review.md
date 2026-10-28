# Headline output and completion-marker review

**No Important or Critical findings.** Reviewed only the requested reporting/test diffs and the summary fields they consume.

- Each CSV and Markdown metric maps to the corresponding saved summary mean/sample-SD field. CSV serialization retains the underlying numeric precision; only Markdown presentation rounds values. Row order puts ridge first and retains all four neural heads.
- Run counts come from the validated summary. The caption distinguishes five deterministic ridge folds from 25 neural fold-by-seed fits and separates these headline values from the pooled seed-averaged ensemble. Mapped PLCC values are means of the independently fitted per-run target-domain mapped correlations, matching the summary implementation; the target-label mapping is disclosed.
- The original-control failure and post-hoc amendment disclosure remain before the table, with historical detail retained in the existing structured outputs. No numerical fitting or metric calculation changes are introduced.
- A failed/incomplete phase-gate sentinel is published before validation. The definitive gate is now written only after summary/accounting/table/report/document outputs succeed. Exceptions during those later writes therefore leave the authoritative completion marker failed instead of retaining a stale pass. Publication remains atomic per file, not a cross-file transaction; consumers should continue treating `phase_gate.json` as the authoritative completion marker.
- The output-failure regression now reaches the OOF/summary path because the fixture supplies nonempty provenance. Its injected headline-write failure checks replacement of an earlier passing gate. The table regression verifies field mapping, run count, rounded presentation and retained CSV precision.

No tests, fits or result regeneration were run. Source, Git and results were not changed; this requested review file is the only write. Actual full Phase 1 report generation remains pending completion of CV/ridge.
