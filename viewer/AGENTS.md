# AGENTS.md

This directory contains the local Next.js viewer for Prism Evals run artifacts.
Use these instructions in addition to the repo root `AGENTS.md`.

## UI Preferences And Persistence

When adding, removing, renaming, or changing viewer controls, always check
whether that UI state should persist across page loads.

Persist user-facing preferences for controls such as:

- Filters, searches, toggles, thresholds, and selected metrics.
- Chart selections and chart configuration.
- Table sorting, column visibility, and column ordering.
- Run, lane, model, score, step, or dataset-field selections.

Do not persist transient UI state such as:

- Loading and error state.
- Open modals, drawers, popovers, or selected detail records.
- Draft input values that are only used to add another saved preference.

Preference persistence lives in `lib/preferences.ts`. When a control is added
or removed, update the relevant versioned preference type, default value,
normalizer, resolver behavior, page wiring, and tests.

Saved references to dynamic run data must be failsafe. Scores, models, steps,
lanes, columns, and charts can disappear between page loads. Keep stale saved
references visible as disabled or unavailable choices when useful, but never
let stale ids drive filtering, chart rendering, table state, or API calls.

When labels are available, store them with saved ids so stale controls can show
plain customer-facing copy such as `Score unavailable` or `Metric unavailable`.
Do not surface implementation details, storage mechanics, or agent instructions
in visible UI copy.

## Testing

For changes that touch persisted viewer state, add or update Vitest coverage in
`test/preferences.test.ts` or nearby viewer tests. Cover both the happy path and
stale-data behavior.

Run these checks before finishing viewer UI changes:

```bash
npm run test
npm run typecheck
```
