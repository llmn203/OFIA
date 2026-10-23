"""Lightweight Stage-5 product-domain adapters."""

from .dual_source_adapter_wrapper import DualSourceAdapterWrapper
from .forcing_adapter import ForcingAdapter
from .ocean_state_adapter import OCEAN_CHANNEL_GROUPS, OceanStateAdapter
from .dual_source_adapter_wrapper_v2 import DualSourceAdapterWrapperV2
from .forcing_adapter_v2 import ForcingAdapterV2
from .lead_conditioned_gate import MonotoneLeadGate
from .ocean_state_adapter_v2 import OceanStateAdapterV2
from .dual_source_adapter_wrapper_phda import ProvenanceHorizonDualSourceAdapterWrapper
from .forcing_adapter_phda import ForcingAdapterPHDA
from .ocean_state_adapter_phda import OceanStateAdapterPHDA
from .provenance_horizon_gate import HorizonForcingGate, ProvenanceHorizonOceanGate
from .dual_source_adapter_wrapper_phda_v2 import FrameAgeHorizonDualSourceAdapterWrapper
from .ocean_state_adapter_phda_v2 import OceanStateAdapterPHDAV2
from .provenance_horizon_gate import FrameAgeHorizonOceanGate
from .dual_interface_spatiotemporal_adapter import (
    DISTAForcingTrajectoryAdapter,
    DISTAOceanHistoryAdapter,
    DISTAOutputTendencyAdapter,
    DualInterfaceSpatiotemporalAdapterWrapper,
)

__all__ = [
    "DualSourceAdapterWrapper",
    "ForcingAdapter",
    "OCEAN_CHANNEL_GROUPS",
    "OceanStateAdapter",
    "DualSourceAdapterWrapperV2",
    "ForcingAdapterV2",
    "MonotoneLeadGate",
    "OceanStateAdapterV2",
    "ProvenanceHorizonDualSourceAdapterWrapper",
    "ForcingAdapterPHDA",
    "OceanStateAdapterPHDA",
    "HorizonForcingGate",
    "ProvenanceHorizonOceanGate",
    "FrameAgeHorizonDualSourceAdapterWrapper",
    "OceanStateAdapterPHDAV2",
    "FrameAgeHorizonOceanGate",
    "DISTAForcingTrajectoryAdapter",
    "DISTAOceanHistoryAdapter",
    "DISTAOutputTendencyAdapter",
    "DualInterfaceSpatiotemporalAdapterWrapper",
]
