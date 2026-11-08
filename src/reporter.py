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
            net_taken = prof.items_taken_count if prof else max(0, picks - putbacks)

            if net_taken > 0:
                took_str = f"YES ({net_taken})"
            elif putbacks > 0:
                took_str = "NO (Put back / Returned)"
            else:
                took_str = "NO"

            rows.append({
                "shopper_id": tid,
                "role": role_val,
                "action": action_val,
                "items_picked": picks,
                "items_put_back": putbacks,
                "net_items_taken": net_taken,
                "took_item": took_str,
                "first_frame": s.first_frame,
                "last_frame": s.last_frame,
                "total_duration_sec": round(s.total_dwell_seconds(engine.fps), 1),
                "highest_state": s.highest_state.value,
                "primary_shelf": s.primary_shelf or "Grocery_Shelf",
                "interactions_count": s.interaction_count,
            })

        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "shopper_sessions.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def export_kpi_json(self, engine: RetailAnalyticsEngine) -> Path:
        """Export executive KPI metrics to JSON."""
        kpis = engine.get_kpis()
        json_path = self.output_dir / "zone_kpis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(kpis, f, indent=2)
        return json_path

    def export_heatmap_image(self, engine: RetailAnalyticsEngine, reference_frame: np.ndarray) -> Path:
        """Save high-resolution shopper dwell heatmap."""
