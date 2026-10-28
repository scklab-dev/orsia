"""Compatibility imports; HITRAN access lives in services.hitran."""

from ..services.hitran import fetch_hitran_lines

__all__ = ["fetch_hitran_lines"]
