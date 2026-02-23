"""Re-identification system for cross-stream pet tracking."""

from .tracker import ReIDTracker
from .extractor import HybridReID, ColorHistogramReID, create_reid_extractor
from .lightweight import AdaptiveReID, FastHistogramReID, create_lightweight_reid
from .global_id import GlobalIDManager
