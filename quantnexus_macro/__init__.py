"""
QuantNexus-Macro
================
Institutional-grade Cross-Asset Macro Transmission & Real-Time Dislocation Engine.
Engineered for quantitative proprietary trading desks, macro funds, and rates/FX market makers.
"""

from quantnexus_macro.config import (
    AssetClass,
    MacroEventType,
    MacroRegime,
    DEFAULT_ASSET_UNIVERSE,
)
from quantnexus_macro.engine import (
    QuantNexusEngine,
    MacroAnalysisPackage,
    build_macro_tracker,
)
from quantnexus_macro.core.streaming import (
    StreamingDislocationBuffer,
    StreamingTick,
    RealTimeDislocationAlert,
)

__version__ = "1.0.0"
__author__ = "QuantNexus Quantitative Research & Engineering"

__all__ = [
    "AssetClass",
    "MacroEventType",
    "MacroRegime",
    "DEFAULT_ASSET_UNIVERSE",
    "QuantNexusEngine",
    "MacroAnalysisPackage",
    "build_macro_tracker",
    "StreamingDislocationBuffer",
    "StreamingTick",
    "RealTimeDislocationAlert",
]
