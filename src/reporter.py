"""
OmniRetail - Reporter Module
Exports structured analytics:
- shopper_sessions.csv (Individual customer timeline)
- zone_kpis.json (Zone & Conversion KPIs)
- conversion_funnel.png (Graphic funnel diagram)
- aisle_heatmap.png (Composite store footprint heatmap)
"""

import json
from pathlib import Path
from typing import Dict, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analytics_engine import RetailAnalyticsEngine


class RetailReporter:
    """Handles persistence of retail analytics reports and visuals."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        engine: RetailAnalyticsEngine,
        reference_frame: np.ndarray,
        profiles: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        """Generate and save all reports."""
        paths = {}
        paths["csv"] = self.export_shopper_csv(engine, profiles=profiles)
        paths["json"] = self.export_kpi_json(engine)
        paths["funnel"] = self.export_conversion_funnel(engine)
        paths["heatmap"] = self.export_heatmap_image(engine, reference_frame)
        return paths

    def export_shopper_csv(self, engine: RetailAnalyticsEngine, profiles: Optional[Dict] = None) -> Path:
        """Export tabular summary of all shopper sessions."""
        rows = []
        for tid, s in engine.sessions.items():
            prof = profiles.get(tid) if profiles else None
            role_val = prof.role.value if prof else "CUSTOMER"
            action_val = prof.current_action.value if prof else s.highest_state.value
            picks = prof.items_picked_count if prof else s.interaction_count
            putbacks = prof.items_putback_count if prof else 0
