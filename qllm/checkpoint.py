"""Resumable CSV output for long sweeps.

A stage run is hours of compute. Accumulating rows in memory and writing once at
the end means a crash, a timeout or a Ctrl-C at hour thirteen of fourteen loses
all of it, so every row is written and flushed as it is produced, and a re-run
skips the configurations the file already holds. This is the same
skip-if-present contract ``hybrid_sweep.py`` already uses, factored out.

Two things make a resume trustworthy rather than merely convenient:

* **A configuration fingerprint.** Resuming into a file produced with different
  hyperparameters would silently blend incompatible measurements into one table.
  The settings that change what a row *means* are hashed into a sidecar, and a
  mismatch refuses to resume instead of appending.
* **Crash-tolerant reading.** A process killed mid-write leaves a truncated final
  line. It is dropped on load rather than parsed into a row of nonsense.

    ck = ResumableCSV(path, key_fields=("layer_type", "chi"), config={...})
    for job in jobs:
        if ck.done(job):        # already in the file from an earlier run
            continue
        ck.add(measure(job))
    ck.close()
    analyse(ck.rows)            # loaded + new, in one list
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def _fingerprint(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ResumableCSV:
    """Append-as-you-go CSV that can pick up where an interrupted run stopped."""

    def __init__(self, path, key_fields, config=None, resume: bool = True,
                 numeric=()):
        self.path = Path(path)
        self.key_fields = tuple(key_fields)
        self.config = dict(config or {})
        self.numeric = tuple(numeric)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        self.rows: list[dict] = []
        self._seen: set[tuple] = set()
        self._fh = None
        self._writer = None
        self.n_resumed = 0

        if resume and self.path.exists():
            self._load()
        elif self.path.exists():
            self.path.unlink()
            self.meta_path.unlink(missing_ok=True)

    # -- reading ------------------------------------------------------------
    def _load(self):
        stored = None
        if self.meta_path.exists():
            try:
                stored = json.loads(self.meta_path.read_text()).get("fingerprint")
            except (OSError, ValueError):
                stored = None
        mine = _fingerprint(self.config)
        if stored is not None and stored != mine:
            raise SystemExit(
                f"{self.path} was written with a different configuration "
                f"(fingerprint {stored} vs {mine}).\nResuming would mix "
                f"incompatible measurements into one table. Move it aside, or "
                f"re-run with --no-resume to start over.")
        with open(self.path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        # A process killed mid-write leaves a short final line; drop it.
        if rows and any(v is None for v in rows[-1].values()):
            rows.pop()
        for r in rows:
            for k in self.numeric:
                if k in r and r[k] != "":
                    try:
                        r[k] = float(r[k]) if "." in r[k] or "e" in r[k].lower() \
                            else int(r[k])
                    except ValueError:
                        pass
            self.rows.append(r)
            self._seen.add(self._key(r))
        self.n_resumed = len(self.rows)
        if self.n_resumed:
            print(f"  resuming {self.path}: {self.n_resumed} row(s) already done")

    def _key(self, row):
        return tuple(str(row.get(f, "")) for f in self.key_fields)

    # -- writing ------------------------------------------------------------
    def done(self, **key) -> bool:
        """Is this configuration already in the file?"""
        return tuple(str(key.get(f, "")) for f in self.key_fields) in self._seen

    def add(self, row: dict):
        """Write one row immediately, flushed, so a crash keeps everything before it."""
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists() or self.path.stat().st_size == 0
            self._fh = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row))
            if new:
                self._writer.writeheader()
            self.meta_path.write_text(json.dumps(
                {"fingerprint": _fingerprint(self.config), "config": self.config},
                indent=2, default=str))
        self._writer.writerow(row)
        self._fh.flush()
        self.rows.append(row)
        self._seen.add(self._key(row))

    def close(self):
        if self._fh is not None:
            self._fh.close()
        self._fh, self._writer = None, None
        print(f"\n{len(self.rows)} row(s) in {self.path.resolve()}"
              + (f" ({self.n_resumed} resumed, "
                 f"{len(self.rows) - self.n_resumed} new)"
                 if self.n_resumed else ""))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
