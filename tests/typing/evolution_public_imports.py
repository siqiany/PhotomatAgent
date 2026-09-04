"""Static regression fixture for lazy public evolution/loop facades."""

from photomatagent.scientific.evolution import EpisodeRecord
from photomatagent.scientific.loop import ScientificLoopSummary


reveal_type(EpisodeRecord)


def reveal_episode_summary(episode: EpisodeRecord) -> None:
    reveal_type(episode.summary)
    summary: ScientificLoopSummary | None = episode.summary
    del summary
