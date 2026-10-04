# Reference fronts (`.KBP`), by version

The instances without a published optimum are measured against the **best front known to this project**:
the non-dominated union of every run of every campaign, with each permutation verified by recomputing its
cost against the instance. Those fronts live here, one directory per version:

```
reference/
  v0.1/   KC20-*.KBP, KC30-*.KBP, summary.json
  v0.2/   the same files, plus the solutions found afterwards
  ...
```

The instances with a published optimum keep using it: `mQAPData/<instance>.PO` is third-party data, it is
never versioned here and never modified.

## Why directories and not git history

A measurement is only meaningful against a named front. Keeping every version on disk lets a table
published months ago stay readable — it says which directory it was measured against — while new runs
keep improving the best known front on the same branch, with no checkout needed to read the old one.

## The rule

1. **A new solution that the current front does not dominate is always added.** Finding one does not
   invalidate the earlier measurements: it means the front has improved, and the earlier numbers remain
   valid *against the version they name*.
2. **A published version is never edited.** Adding solutions creates the next version, which starts as a
   copy of the previous one.
3. **Every table in the documentation says which version it was measured against.** A share of
   hypervolume or of coverage only means something next to the front it was computed from.

To add what a set of runs found:

```
python scripts/compare_versions.py KC20-2fl-1rl "label=path/to/runs.txt" --update-reference
```

It verifies every candidate permutation against the instance, writes the next version directory with the
files that change and copies over the ones that do not, and prints the factor that converts a share
published against the previous version into the new scale.

## Versions

| Version | Date | What it adds |
|---|---|---|
| `v0.1` | 2026-09-26 | The convergence campaign run with the pair traversal of the original version: 7 instances, from 8 points (KC20-2fl-2uni) to 16,989 (KC30-3fl-1rl). |
| `v0.2` | 2026-10-04 | 7 solutions found by the greedy-rate experiment on KC20 with P = 65536, on KC20-2fl-1rl and KC20-2fl-3uni. |
