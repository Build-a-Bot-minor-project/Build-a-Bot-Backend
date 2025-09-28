"""
BuildABot Agents Module

This module contains agent configurations, presets, and utilities
for managing AI agents in the BuildABot platform.
"""

from .preset_agents import PRESET_AGENTS, get_preset_agent_config, get_all_preset_names

__all__ = [
    "PRESET_AGENTS",
    "get_preset_agent_config", 
    "get_all_preset_names"
]
