"""Tidy, append-only storage for optimisation traces.

One long-format CSV per experiment rather than one file per run: a stage sweep
has thousands of runs, and thousands of little files are neither neat nor
groupable. Every row carries the full configuration key, so the file is
self-describing and a single pandas/csv read answers "show me every gradient
curve for this ansatz".

Columns
    run_id      stable key for one optimisation (config fields joined)
    ... the configuration fields the caller passes (layer, chi, ansatz, ...)
    phase       'explicit' (environment sweep) or 'gradient' (Adam)
    iter        1-based, WITHIN the phase
    objective   what ``loss`` is -- 'retained-weight', 'relative-error',
                'disentangle-loss' or 'activation-mse'
    loss        the quantity being minimised, lower is better
    score       higher-is-better companion where one exists (retained weight);
                NaN on gradient rows, which have no such quantity
    best        1 if this iteration was a new best-so-far
"""

from __future__ import annotations

import csv
from pathlib import Path

FIELDS = ("phase", "iter", "objective", "loss", "score", "best")


class TraceWriter:
    """Appends trace rows to one CSV, writing the header on first use.

    Opened lazily so that a run which never produces a trace leaves no stray
    empty file behind.
    """

    def __init__(self, path, every: int = 1, append: bool = False):
        self.path = Path(path) if path else None
        self.every = max(1, int(every))
        self.append = bool(append)
        self._fh = None
        self._writer = None
        self.n_rows = 0
        self.n_runs = 0

    def __bool__(self):
        return self.path is not None

    def add(self, result, **key):
        """Record a :class:`DisentangleResult`'s trace (see :meth:`add_rows`)."""
        return self.add_rows(getattr(result, "trace", None), **key)

    def add_rows(self, trace, **key):
        """Record one :class:`DisentangleResult`'s trace under a configuration key.

        ``every`` thins the GRADIENT phase only -- a 150-step Adam run is the bulk
        of the rows while the handful of sweep iterations are all worth keeping.
        The first and last iteration of a phase are always kept so the endpoints
        of every curve survive thinning.
        """
        if self.path is None or not trace:
            return
        rows = []
        by_phase = {}
        for t in trace:
            by_phase.setdefault(t["phase"], []).append(t)
        for phase, items in by_phase.items():
            last = len(items)
            for t in items:
                if (phase == "gradient" and self.every > 1
                        and t["iter"] % self.every
                        and t["iter"] not in (1, last)):
                    continue
                rows.append(t)
        if not rows:
            return
        run_id = "|".join(f"{v}" for v in key.values())
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Append when the surrounding run is resuming, so the traces of the
            # work already done are not thrown away by the run that finishes it.
            fresh = not (self.append and self.path.exists()
                         and self.path.stat().st_size)
            self._fh = open(self.path, "w" if fresh else "a", newline="")
            self._writer = csv.DictWriter(
                self._fh, fieldnames=["run_id", *key.keys(), *FIELDS])
            if fresh:
                self._writer.writeheader()
        for t in rows:
            self._writer.writerow({"run_id": run_id, **key,
                                   **{f: t[f] for f in FIELDS}})
        self._fh.flush()          # a crash keeps every trace written before it
        self.n_rows += len(rows)
        self.n_runs += 1

    def close(self):
        if self._fh is not None:
            self._fh.close()
            print(f"  wrote {self.n_rows} trace rows from {self.n_runs} run(s) "
                  f"to {self.path.resolve()}")
        self._fh, self._writer = None, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_traces(path):
    """Load a trace CSV back as a list of dicts with numbers parsed."""
    out = []
    for r in csv.DictReader(open(path)):
        r["iter"] = int(r["iter"])
        r["loss"] = float(r["loss"])
        r["score"] = float(r["score"])
        r["best"] = int(r["best"])
        out.append(r)
    return out
