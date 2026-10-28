"""Seeded parent selection from trusted archive metadata."""
from __future__ import annotations

import random
from collections.abc import Mapping

from photomatagent.scientific.evolution.qd.models import ArchiveEntry, WorkflowGenome


def select_parents(
    entries: tuple[ArchiveEntry, ...], genomes_by_id: Mapping[str, WorkflowGenome],
    seed: int, limit: int, *, task_family: str | None = None,
    applicability: str | None = None,
) -> tuple[WorkflowGenome, ...]:
    selected = select_parent_entries(entries, genomes_by_id, seed, limit,
                                     task_family=task_family, applicability=applicability)
    return tuple(genomes_by_id[entry.genome_id] for entry in selected)


def select_parent_entries(
    entries: tuple[ArchiveEntry, ...], genomes_by_id: Mapping[str, WorkflowGenome],
    seed: int, limit: int, *, task_family: str | None = None,
    applicability: str | None = None,
) -> tuple[ArchiveEntry, ...]:
    if limit <= 0:
        return ()
    eligible = [entry for entry in entries if (
        (genome := genomes_by_id.get(entry.genome_id)) is not None
        and genome.genome_id == entry.genome_id
        and (task_family is None or genome.task_family == task_family)
        and (applicability is None or applicability in genome.applicability)
    )]
    eligible.sort(key=lambda item: (item.use_count, item.last_used_at_ns, item.cell_key))
    rng = random.Random(seed)
    selected: list[ArchiveEntry] = []
    seen: set[str] = set()
    while eligible and len(selected) < limit:
        entry = eligible.pop(rng.randrange(min(4, len(eligible))))
        if entry.genome_id in seen:
            continue
        seen.add(entry.genome_id)
        selected.append(entry)
    return tuple(selected)


__all__ = ["select_parent_entries", "select_parents"]
