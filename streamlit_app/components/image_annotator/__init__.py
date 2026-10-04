"""
Interactive Click Annotator Streamlit Custom Component.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_component_func = components.declare_component("image_annotator", path=_COMPONENT_DIR)


def image_annotator(
    image_data: str,
    orig_width: int,
    orig_height: int,
    spaces: List[Dict[str, Any]],
    current_points: List[List[int]],
    current_space_id: str,
    key: Optional[str] = None,
    default: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Render the interactive canvas and capture click events from the user.

    Args:
        image_data: Base64 data URL of the image.
        orig_width: Original image pixel width.
        orig_height: Original image pixel height.
        spaces: List of completed parking space definitions.
        current_points: List of in-progress corner points [[x, y], ...].
        current_space_id: Active space ID label (e.g. 'P1').
        key: Streamlit widget key.
        default: Default return value.

    Returns:
        Dict containing clicked coordinates {'x': int, 'y': int, 'click_id': float} or default.
    """
    # Ensure component is registered in Streamlit runtime component_registry
    try:
        from streamlit.runtime import get_instance
        runtime = get_instance()
        if runtime and hasattr(runtime, "component_registry"):
            runtime.component_registry.register_component(_component_func)
    except Exception:
        pass

    return _component_func(
        image_data=image_data,
        orig_width=orig_width,
        orig_height=orig_height,
        spaces=spaces,
        current_points=current_points,
        current_space_id=current_space_id,
        key=key,
        default=default,
    )
