"""Optional, bounded discovery and installation planning for capabilities.

Importing this package performs no network access and loads no optional
network client. Providers are constructed only when explicitly configured.
"""

from photomatagent.capability_acquisition.models import (
    DiscoveryResult,
    InstallPlan,
    PackageCandidate,
    WheelArtifact,
)

__all__ = ["DiscoveryResult", "InstallPlan", "PackageCandidate", "WheelArtifact"]
