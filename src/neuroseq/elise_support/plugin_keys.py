"""Ids and widget attributes shared by the ELiSe workflows and their plugins.

Core code must not import from ``resources/analysis_plugins``; the plugins
import these names from here instead.
"""

from __future__ import annotations

TARGET_MSE_PLUGIN_ID = "elise_target_mse"
# Attributes the replay analysis sets on the analysis widget for the plugin.
TARGET_ATTR = "_elise_target_rate"
TARGET_DT_ATTR = "_elise_target_dt_s"

LEARNING_CURVES_PLUGIN_ID = "elise_learning_curves"
# Attribute the protocol workflow sets on the widget to hand over the curves.
CURVES_ATTR = "_elise_protocol_curves"
