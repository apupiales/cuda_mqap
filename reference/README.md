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

A new version copies the instances it does not change, so it reads as a complete set, and that costs
almost nothing in the repository: the copies are byte-identical, and git stores one blob for all of them.

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
published against the previous version into the new scale. `--into v0.3` adds to a version that is already
open, which is how the several instances of one experiment share a version.

To build the front of an instance that has none yet, or to rebuild one from a whole campaign:

```
python scripts/build_reference.py KC20-2fl-2rl "results/grid/KC20-2fl-2rl/*.txt" --into v0.3
```

It takes the non-dominated union of every run, verifies the permutation of each survivor, and writes it
into that version. The objective vectors decide who survives before any permutation is verified, which is
what makes a front of hundreds of runs affordable to build.

## Versions

| Version | Date | What it adds |
|---|---|---|
| `v0.1` | 2026-09-26 | The convergence campaign run with the pair traversal of the original version: 7 instances, from 8 points (KC20-2fl-2uni) to 16,989 (KC30-3fl-1rl). |
| `v0.2` | 2026-10-04 | 7 solutions found by the greedy-rate experiment on KC20 with P = 65536, on KC20-2fl-1rl and KC20-2fl-3uni. |
| `v0.3` | 2026-10-05 | The grid of population × greedy configuration, 690 cells of 10 runs. It gives a first front to the 8 instances that had none (KC20-2fl-2rl, 3rl, 4rl, 5rl, KC30-2fl-1rl, KC30-3fl-2rl, 3rl, 3uni) and improves 4 of the 7 that had one: KC30-3fl-1rl 16,989 → 17,097, KC30-3fl-1uni 3448 → 3562, KC30-3fl-2uni 821 → 847, KC20-2fl-3uni 241 → 243. |
| `v0.4` | 2026-10-05 | 438 solutions found by one run of the default call on each of the 23 instances (`scripts/run_front_plot.ps1`, seed 20261005), on seven of them: KC20-2fl-3rl 215 → 216, KC20-2fl-4rl 99 → 100, KC30-2fl-1rl 251 → 251 (3 in, 3 out), KC30-3fl-1uni 3562 → 3567, KC30-3fl-2rl 13,388 → 13,421, KC30-3fl-3rl 26,219 → 26,409, KC30-3fl-3uni 4181 → 4221. The 168 points of `v0.3` they dominate leave the front. |

Every instance without a published optimum now has a front: 15 of the 23 in `mQAPData`, the other 8 being
the KC10 ones, which have theirs published.
