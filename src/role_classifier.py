"""
OmniRetail - Role & Action Classifier Module with Pick vs. Put-Back Analytics
Accurately distinguishes between:
- Role: CUSTOMER / SHOPPER vs. MART STAFF / SUPERVISOR
- Micro-Actions: WALKING, BROWSING SHELF, REACHING / PICKING, INSPECTING ITEM (HELD), PUT BACK ON SHELF, MONITORING AISLE
- Pick vs. Put-Back Verification:
  - Did they pick an item? (YES / NO with pick count)
  - Did they put it back on the shelf? (YES / NO with put-back count)
  - Net Items Taken: max(0, picks - putbacks)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .config import ZoneDefinition
from .pose_analyzer import ShopperPose
from .tracker import ShopperDetection


class PersonRole(Enum):
    CUSTOMER = "CUSTOMER"
    SUPERVISOR = "SUPERVISOR / STAFF"


class DetailedAction(Enum):
    WALKING = "WALKING IN AISLE"
    BROWSING = "BROWSING SHELF"
    REACHING = "REACHING FOR SHELF"
    PICKED_ITEM = "PICKED ITEM FROM SHELF"
    ITEM_PICKED = "PICKED ITEM FROM SHELF"
    INSPECTING_ITEM = "INSPECTING ITEM (HELD)"
    PUT_BACK = "PUT BACK ON SHELF"
    MONITORING = "MONITORING AISLE"


@dataclass
class PersonBehaviorProfile:
    track_id: int
    role: PersonRole = PersonRole.CUSTOMER
    current_action: DetailedAction = DetailedAction.WALKING
    items_picked_count: int = 0
    items_putback_count: int = 0
    items_taken_count: int = 0       # Net items taken: max(0, picked - putback)
    is_holding_item: bool = False
    holding_item_frames: int = 0
    reach_frames_streak: int = 0
    in_retract_phase: bool = False
