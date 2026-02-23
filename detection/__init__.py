"""Behavior detection modules for pet tracking."""

from .fight import detect_fight
from .inert import detect_inert
from .sleep import detect_sleep
from .eat import detect_eat
from .bathroom import detect_bathroom
from .active import detect_active
from .escape import detect_escape, polygon_selector
