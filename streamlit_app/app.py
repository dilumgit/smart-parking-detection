"""
Smart Parking Detection AI - Streamlit Web Dashboard.

This frontend interacts with the FastAPI backend over HTTP to:
1. Check real-time parking space occupancy from uploaded images.
2. Interactively configure and persist named parking area polygon layouts via direct canvas clicks.
"""

import base64
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(
    page_title="Smart Parking Detection AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern clean UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .badge-success {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Import custom interactive canvas annotator component
from streamlit_app.components.image_annotator import image_annotator


# =====================================================================
# API CLIENT HELPER FUNCTIONS
# =====================================================================

def check_api_health(base_url: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Check the connection and health status of the FastAPI backend.

    Args:
        base_url: Base URL of the FastAPI backend (e.g. 'http://127.0.0.1:8000').

    Returns:
        Tuple containing (is_healthy, response_data, error_message).
    """
    health_endpoint = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(health_endpoint, timeout=3.0)
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, f"HTTP {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return False, None, "Connection refused. Ensure the FastAPI backend is running."
    except requests.exceptions.Timeout:
        return False, None, "Request timed out connecting to FastAPI backend."
    except Exception as exc:
        return False, None, f"Unexpected error: {str(exc)}"


def fetch_saved_areas(base_url: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch the list of all saved parking area summaries from FastAPI.

    Args:
        base_url: Base URL of the FastAPI backend.

    Returns:
        Tuple containing (areas_list, error_message).
    """
    areas_endpoint = f"{base_url.rstrip('/')}/api/v1/parking/areas"
    try:
        response = requests.get(areas_endpoint, timeout=5.0)
        if response.status_code == 200:
            return response.json(), None
        return [], f"Failed to fetch areas (HTTP {response.status_code}): {response.text}"
    except requests.exceptions.ConnectionError:
        return [], "Unable to connect to FastAPI backend to fetch parking areas."
    except requests.exceptions.Timeout:
        return [], "Request timed out fetching saved parking areas."
    except Exception as exc:
        return [], f"Error fetching saved parking areas: {str(exc)}"


def save_parking_area_api(
    base_url: str,
    area_payload: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Send a newly created parking area configuration to FastAPI for persistence.

    Args:
        base_url: Base URL of the FastAPI backend.
        area_payload: Dict containing parking_area_id, name, reference_image, and spaces.

    Returns:
        Tuple containing (saved_area_dict, error_message).
    """
    areas_endpoint = f"{base_url.rstrip('/')}/api/v1/parking/areas"
    try:
        response = requests.post(areas_endpoint, json=area_payload, timeout=10.0)
        if response.status_code in (200, 201):
            return response.json(), None

        # Parse error detail
        try:
            err_json = response.json()
            detail = err_json.get("detail", response.text)
        except Exception:
            detail = response.text
        return None, f"API Error (HTTP {response.status_code}): {detail}"

    except requests.exceptions.ConnectionError:
        return None, "Unable to connect to FastAPI backend. Ensure server is running."
    except requests.exceptions.Timeout:
        return None, "The save request timed out connecting to FastAPI."
    except Exception as exc:
        return None, f"Unexpected error while saving parking area: {str(exc)}"


def send_analyze_request(
    base_url: str,
    file_bytes: bytes,
    filename: str,
    parking_area_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Send an uploaded parking image to the FastAPI backend for analysis.

    Args:
        base_url: Base URL of the FastAPI backend.
        file_bytes: Raw binary bytes of the image.
        filename: Name of the uploaded file.
        parking_area_id: Optional ID of the saved parking area to use.

    Returns:
        Tuple containing (analysis_response_dict, error_message).
    """
    analyze_endpoint = f"{base_url.rstrip('/')}/api/v1/parking/analyze"
    files = {"file": (filename, file_bytes, "image/jpeg")}
    params = {}
    if parking_area_id and parking_area_id != "default":
        params["parking_area_id"] = parking_area_id

    try:
        response = requests.post(analyze_endpoint, files=files, params=params, timeout=25.0)
        if response.status_code == 200:
            return response.json(), None

        try:
            err_json = response.json()
            detail = err_json.get("detail", response.text)
        except Exception:
            detail = response.text
        return None, f"API Error (HTTP {response.status_code}): {detail}"

    except requests.exceptions.ConnectionError:
        return None, "Unable to connect to the FastAPI backend. Make sure the API server is running."
    except requests.exceptions.Timeout:
        return None, "The analysis request timed out. The server took too long to respond."
    except Exception as exc:
        return None, f"An unexpected error occurred while communicating with the API: {str(exc)}"


# =====================================================================
# VISUAL RENDERING FUNCTIONS
# =====================================================================

def render_spaces_preview(
    image_bgr: np.ndarray,
    spaces: List[Dict[str, Any]],
    current_points: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Render completed space polygons and in-progress points over an image.

    Args:
        image_bgr: BGR format OpenCV image.
        spaces: List of space dicts with 'space_id' and 'polygon'.
        current_points: Optional list of in-progress (x, y) points.

    Returns:
        RGB numpy array ready for Streamlit st.image display.
    """
    canvas = image_bgr.copy()
    img_h, img_w = canvas.shape[:2]

    # 1. Draw all completed parking spaces
    for space in spaces:
        space_id = space.get("space_id") or space.get("id", "P")
        pts = np.array(space["polygon"], dtype=np.int32).reshape((-1, 1, 2))

        # Green polygon boundary
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 220, 0), thickness=2)

        # Semi-transparent fill overlay
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 220, 0))
        cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)

        # Space label badge at centroid
        moments = cv2.moments(pts)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
        else:
            cX, cY = int(space["polygon"][0][0]), int(space["polygon"][0][1])

        label = str(space_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        lx1 = max(0, cX - tw // 2 - 4)
        ly1 = max(0, cY - th // 2 - 4)
        lx2 = min(img_w, lx1 + tw + 8)
        ly2 = min(img_h, ly1 + th + 8)

        cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            label,
            (lx1 + 4, ly2 - 4),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    # 2. Draw in-progress points if any
    if current_points:
        for i, pt in enumerate(current_points):
            cv2.circle(canvas, pt, 5, (0, 0, 255), -1)
            cv2.circle(canvas, pt, 7, (255, 255, 255), 1)
            if i > 0:
                cv2.line(canvas, current_points[i - 1], pt, (0, 165, 255), 2)

    # Convert BGR to RGB for Streamlit display
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# =====================================================================
# UI COMPONENTS
# =====================================================================

def display_summary_metrics(summary: Dict[str, Any]) -> None:
    """Render high-level occupancy summary metrics using Streamlit metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Spaces",
            value=summary.get("total_spaces", 0),
        )
    with col2:
        st.metric(
            label="Occupied Spaces",
            value=summary.get("occupied_spaces", 0),
            delta=f"{summary.get('occupied_spaces', 0)} vehicles",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="Empty Spaces",
            value=summary.get("empty_spaces", 0),
            delta=f"{summary.get('empty_spaces', 0)} free",
            delta_color="normal",
        )
    with col4:
        occupancy_rate = summary.get("occupancy_rate", 0.0)
        st.metric(
            label="Occupancy Rate",
            value=f"{occupancy_rate:.1f}%",
        )


def display_parking_spaces_table(parking_spaces: List[Dict[str, Any]]) -> None:
    """Format and display detailed parking space occupancy data in a structured table."""
    table_rows = []
    for space in parking_spaces:
        status = space.get("status", "EMPTY")
        status_display = "🔴 OCCUPIED" if status == "OCCUPIED" else "🟢 EMPTY"

        vehicle_class = space.get("vehicle_class")
        confidence = space.get("confidence")

        conf_str = f"{confidence:.2f}" if confidence is not None else "—"
        class_str = vehicle_class.capitalize() if vehicle_class else "—"

        table_rows.append(
            {
                "Space ID": space.get("space_id", "N/A"),
                "Status": status_display,
                "Vehicle Count": space.get("vehicle_count", 0),
                "Vehicle Class": class_str,
                "Confidence": conf_str,
            }
        )

    df = pd.DataFrame(table_rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Space ID": st.column_config.TextColumn("Space ID", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Vehicle Count": st.column_config.NumberColumn("Vehicles", width="small"),
            "Vehicle Class": st.column_config.TextColumn("Detected Class", width="medium"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
        },
    )


# =====================================================================
# MODE: PARKING AREA SETUP (PHASE 10.1B.1 - CLICKABLE CANVAS)
# =====================================================================

def render_parking_area_setup(api_url: str, is_healthy: bool) -> None:
    """Render the interactive Parking Area Setup interface with direct image clicking."""
    st.subheader("📐 Parking Area Setup & Configuration")
    st.caption("Click 4 corners directly on the reference image to mark each parking space polygon.")

    # Initialize session state collections for setup
    if "setup_spaces" not in st.session_state:
        st.session_state["setup_spaces"] = []
    if "setup_points" not in st.session_state:
        st.session_state["setup_points"] = []
    if "last_processed_click_id" not in st.session_state:
        st.session_state["last_processed_click_id"] = None

    # Step 1: Parking Area Metadata
    st.markdown("#### 1. Area Identification")
    col_name, col_id = st.columns(2)

    with col_name:
        area_name = st.text_input(
            "Parking Area Name",
            value=st.session_state.get("setup_area_name", "Main Parking Lot"),
            help="Human-readable title for this parking lot or camera viewpoint",
            key="input_area_name",
        )

    # Auto-generate default ID based on name
    suggested_id = re.sub(r"[^a-zA-Z0-9_-]", "_", area_name.strip().lower())
    if not suggested_id:
        suggested_id = "parking_area_002"

    with col_id:
        area_id = st.text_input(
            "Parking Area ID",
            value=st.session_state.get("setup_area_id", suggested_id),
            help="Unique identifier (alphanumeric, hyphens, and underscores only)",
            key="input_area_id",
        )

    # Step 2: Upload Reference Image
    st.markdown("#### 2. Upload Reference Image")
    ref_image_file = st.file_uploader(
        "Upload Reference Image of Parking Area (JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="setup_image_uploader",
        help="Upload the fixed camera reference image on which parking spaces will be marked.",
    )

    if ref_image_file is None:
        st.info("👆 Please upload a reference image to begin defining parking spaces.")
        return

    # Check if a new image was uploaded; reset drawing state if changed
    if st.session_state.get("active_ref_image_name") != ref_image_file.name:
        st.session_state["active_ref_image_name"] = ref_image_file.name
        st.session_state["setup_spaces"] = []
        st.session_state["setup_points"] = []
        st.session_state["last_processed_click_id"] = None

    file_bytes = ref_image_file.getvalue()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Failed to decode the uploaded image file. Please provide a valid JPEG or PNG.")
        return

    img_h, img_w = image_bgr.shape[:2]

    # Convert image to Base64 data URL for component
    encoded_b64 = base64.b64encode(file_bytes).decode("utf-8")
    mime_type = "image/png" if ref_image_file.name.lower().endswith(".png") else "image/jpeg"
    image_data_url = f"data:{mime_type};base64,{encoded_b64}"

    # Step 3: Interactive Clickable Canvas
    st.markdown("#### 3. Mark Parking Spaces on Image")

    current_space_num = len(st.session_state["setup_spaces"]) + 1
    current_space_id = f"P{current_space_num}"
    current_point_count = len(st.session_state["setup_points"])

    # Status Bar & Controls
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns([1.5, 1.2, 1.2, 1.2])

    with col_stat1:
        st.markdown(
            f"**Current Space:** `{current_space_id}` &nbsp;|&nbsp; "
            f"**Points:** `{current_point_count} / 4`"
        )

    with col_stat2:
        undo_clicked = st.button(
            "↩️ Undo Last Point",
            use_container_width=True,
            disabled=(len(st.session_state["setup_points"]) == 0),
            help="Remove the most recently clicked corner point for the in-progress space",
        )
        if undo_clicked and st.session_state["setup_points"]:
            st.session_state["setup_points"].pop()
            st.rerun()

    with col_stat3:
        delete_space_clicked = st.button(
            "🗑️ Delete Last Space",
            use_container_width=True,
            disabled=(len(st.session_state["setup_spaces"]) == 0),
            help="Remove the most recently completed parking space and return to editing it",
        )
        if delete_space_clicked and st.session_state["setup_spaces"]:
            st.session_state["setup_spaces"].pop()
            st.rerun()

    with col_stat4:
        reset_clicked = st.button(
            "🔄 Reset All",
            use_container_width=True,
            disabled=(len(st.session_state["setup_spaces"]) == 0 and len(st.session_state["setup_points"]) == 0),
            help="Clear all current points and completed spaces",
        )
        if reset_clicked:
            st.session_state["setup_spaces"] = []
            st.session_state["setup_points"] = []
            st.session_state["last_processed_click_id"] = None
            st.rerun()

    # Render custom HTML5 click canvas component
    click_data = image_annotator(
        image_data=image_data_url,
        orig_width=img_w,
        orig_height=img_h,
        spaces=st.session_state["setup_spaces"],
        current_points=st.session_state["setup_points"],
        current_space_id=current_space_id,
        key=f"annotator_{ref_image_file.name}",
        default=None,
    )

    # Process incoming click event from canvas
    if click_data and isinstance(click_data, dict) and "click_id" in click_data:
        click_id = click_data["click_id"]
        if click_id != st.session_state.get("last_processed_click_id"):
            st.session_state["last_processed_click_id"] = click_id
            clicked_x = int(click_data.get("x", 0))
            clicked_y = int(click_data.get("y", 0))

            # Append new corner point
            st.session_state["setup_points"].append([clicked_x, clicked_y])

            # When 4 points are reached, automatically complete and close the space
            if len(st.session_state["setup_points"]) == 4:
                completed_space_id = f"P{len(st.session_state['setup_spaces']) + 1}"
                st.session_state["setup_spaces"].append({
                    "space_id": completed_space_id,
                    "polygon": list(st.session_state["setup_points"]),
                })
                st.session_state["setup_points"] = []

            st.rerun()

    # Optional Preload Reference Helper (for quick testing with 1.jpg)
    col_preset, col_info = st.columns([2, 3])
    with col_preset:
        if st.button("⚡ Preload Reference Spaces (P1–P4 for 1.jpg)", use_container_width=True):
            st.session_state["setup_spaces"] = [
                {"space_id": "P1", "polygon": [[45, 123], [110, 237], [213, 212], [138, 105]]},
                {"space_id": "P2", "polygon": [[140, 104], [212, 214], [312, 188], [230, 87]]},
                {"space_id": "P3", "polygon": [[231, 87], [312, 189], [406, 166], [319, 68]]},
                {"space_id": "P4", "polygon": [[321, 70], [408, 167], [498, 146], [405, 51]]},
            ]
            st.session_state["setup_points"] = []
            st.rerun()

    with col_info:
        st.caption(f"Original Canvas Resolution: **{img_w} × {img_h} px** | Total Spaces Defined: **{len(st.session_state['setup_spaces'])}**")

    # Display Defined Spaces Summary Table
    if st.session_state["setup_spaces"]:
        st.divider()
        st.markdown(f"##### Defined Parking Spaces ({len(st.session_state['setup_spaces'])})")
        spaces_table_rows = []
        for s in st.session_state["setup_spaces"]:
            spaces_table_rows.append({
                "Space ID": s["space_id"],
                "Corner 1": f"({s['polygon'][0][0]}, {s['polygon'][0][1]})",
                "Corner 2": f"({s['polygon'][1][0]}, {s['polygon'][1][1]})",
                "Corner 3": f"({s['polygon'][2][0]}, {s['polygon'][2][1]})",
                "Corner 4": f"({s['polygon'][3][0]}, {s['polygon'][3][1]})",
            })
        st.dataframe(pd.DataFrame(spaces_table_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Step 4: Save Parking Area to Backend
    st.markdown("#### 4. Save Parking Area")
    save_btn = st.button("💾 Save Parking Area", type="primary", use_container_width=False)

    if save_btn:
        # 1. Validation
        if not area_name or not area_name.strip():
            st.error("Please provide a valid Parking Area Name.")
            return

        if not area_id or not area_id.strip():
            st.error("Please provide a valid Parking Area ID.")
            return

        if not re.match(r"^[a-zA-Z0-9_-]+$", area_id.strip()):
            st.error("Parking Area ID may only contain alphanumeric characters, hyphens, and underscores.")
            return

        if not st.session_state["setup_spaces"]:
            st.error("Please define at least one parking space by clicking 4 corners before saving.")
            return

        for space in st.session_state["setup_spaces"]:
            poly = space.get("polygon", [])
            if len(poly) != 4:
                st.error(f"Space '{space.get('space_id')}' must have exactly 4 points.")
                return

        if not is_healthy:
            st.error(f"Cannot save: FastAPI backend at {api_url} is offline. Please start FastAPI.")
            return

        # 2. Build Payload
        area_payload = {
            "parking_area_id": area_id.strip(),
            "name": area_name.strip(),
            "reference_image": ref_image_file.name,
            "spaces": st.session_state["setup_spaces"],
        }

        # 3. Send HTTP Request to FastAPI
        with st.spinner(f"Saving parking area '{area_id}' to FastAPI backend..."):
            saved_result, save_err = save_parking_area_api(
                base_url=api_url,
                area_payload=area_payload,
            )

            if save_err:
                st.error(f"❌ {save_err}")
            elif saved_result:
                st.success(f"🎉 Parking Area **'{saved_result.get('name')}'** saved successfully!")
                st.info(
                    f"• **Area ID:** `{saved_result.get('parking_area_id')}`\n"
                    f"• **Total Spaces:** `{saved_result.get('total_spaces')}`\n"
                    f"• **Reference Image:** `{saved_result.get('reference_image')}`\n"
                    f"• **Configuration File:** `data/parking_configs/{saved_result.get('parking_area_id')}.json`"
                )
                st.session_state["last_saved_area"] = saved_result


# =====================================================================
# MODE: CHECK PARKING AVAILABILITY
# =====================================================================

def render_check_availability(api_url: str, is_healthy: bool, saved_areas: List[Dict[str, Any]]) -> None:
    """Render the Parking Occupancy Analysis interface."""
    st.subheader("1. 🎯 Select Parking Area & Upload Image")

    col_area, col_upload = st.columns([1, 2])

    with col_area:
        area_options = ["default (Default Parking Config)"]
        area_id_map = {"default (Default Parking Config)": "default"}

        for a in saved_areas:
            label = f"{a['parking_area_id']} — {a['name']} ({a['total_spaces']} spaces)"
            area_options.append(label)
            area_id_map[label] = a["parking_area_id"]

        selected_label = st.selectbox(
            "Active Parking Area Layout",
            options=area_options,
            help="Select the saved parking area polygon layout to analyze against.",
        )
        selected_area_id = area_id_map.get(selected_label, "default")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Parking Lot Image (JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a fixed-camera parking lot photograph for real-time analysis",
        )

    col_img, col_info = st.columns([3, 2])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        with col_img:
            st.image(
                uploaded_file,
                caption=f"Uploaded: {uploaded_file.name} ({len(file_bytes) / 1024:.1f} KB)",
                use_container_width=True,
            )
        with col_info:
            st.info(
                f"Image loaded. Selected Area Layout: `{selected_area_id}`.\n\n"
                "Click **🚀 Analyze Parking Lot** below to run YOLO vehicle detection and space occupancy mapping."
            )
    else:
        st.info("Please upload a parking lot image to begin analysis.")

    st.write("")

    # Action Button
    st.subheader("2. 🔍 Analyze Parking")
    analyze_button = st.button("🚀 Analyze Parking Lot", type="primary", use_container_width=False)

    if analyze_button:
        if uploaded_file is None:
            st.warning("Please upload a parking image first before clicking Analyze.")
        elif not is_healthy:
            st.error(
                "Unable to connect to the FastAPI backend. "
                "Make sure the API server is running at " + api_url
            )
        else:
            with st.spinner("Communicating with FastAPI backend (running YOLO inference & occupancy mapping)..."):
                result, err = send_analyze_request(
                    base_url=api_url,
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    parking_area_id=selected_area_id if selected_area_id != "default" else None,
                )

                if err:
                    st.error(err)
                elif result:
                    st.session_state["last_result"] = result
                    st.session_state["analyzed_filename"] = uploaded_file.name
                    st.session_state["analyzed_area_id"] = selected_area_id
                    st.success("✅ Analysis completed successfully!")

    # Display Analysis Results (from current run or session state)
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        summary = result.get("summary", {})
        parking_spaces = result.get("parking_spaces", [])

        st.divider()

        # Section 3 - Occupancy Summary
        st.subheader("3. 📊 Parking Occupancy Summary")
        display_summary_metrics(summary)

        st.write("")

        # Section 4 - Parking Space Status Table
        st.subheader("4. 🚗 Parking Space Status")
        display_parking_spaces_table(parking_spaces)


# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================

def main():
    # Ensure custom components are registered in Streamlit runtime component_registry
    try:
        from streamlit.runtime import get_instance
        from streamlit_app.components.image_annotator import _component_func
        runtime = get_instance()
        if runtime and hasattr(runtime, "component_registry"):
            runtime.component_registry.register_component(_component_func)
    except Exception:
        pass

    # Header Section
    st.markdown('<div class="main-header">🚗 Smart Parking Detection AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">YOLO-based Vehicle Detection and Real-Time Parking Space Occupancy Analysis</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Sidebar: Backend Configuration & Mode Navigation
    with st.sidebar:
        st.header("⚙️ Backend Configuration")
        default_api_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
        api_url = st.text_input(
            "FastAPI Base URL",
            value=default_api_url,
            help="Base URL of the running FastAPI backend server",
        )

        # Connection Status Check
        is_healthy, health_data, health_err = check_api_health(api_url)
        if is_healthy:
            st.success("🟢 API Connected")
            if health_data:
                st.caption(f"**Service:** `{health_data.get('service', 'smart-parking-ai')}`")
        else:
            st.error("🔴 API Offline")
            if health_err:
                st.caption(f"*{health_err}*")

        st.divider()

        # Mode Selection Navigation
        st.header("🧭 Navigation")
        mode = st.radio(
            "Select Application Mode",
            options=["🚗 Check Parking Availability", "📐 Parking Area Setup"],
            index=0,
            help="Switch between monitoring parking availability and configuring new parking area boundaries.",
        )

        st.divider()

        # Saved Parking Areas Overview in Sidebar
        st.header("📁 Saved Parking Areas")
        saved_areas: List[Dict[str, Any]] = []
        if is_healthy:
            areas, err = fetch_saved_areas(api_url)
            if not err:
                saved_areas = areas
                for a in saved_areas:
                    st.markdown(
                        f"• **{a['name']}** (`{a['parking_area_id']}`)\n"
                        f"  <span class='badge-success'>{a['total_spaces']} spaces</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(f"*{err}*")
        else:
            st.caption("*Connect backend to view saved areas.*")

        st.divider()
        st.markdown("### 📌 System Architecture")
        st.caption(
            "• **Frontend**: Streamlit Dashboard\n"
            "• **Backend**: FastAPI REST API\n"
            "• **AI Model**: Ultralytics YOLO11n\n"
            "• **Geometry**: OpenCV Point-in-Polygon\n"
        )
        st.divider()
        st.caption("Smart Parking AI — Phase 10.1B.1")

    # Render selected mode interface
    if mode == "📐 Parking Area Setup":
        render_parking_area_setup(api_url=api_url, is_healthy=is_healthy)
    else:
        render_check_availability(api_url=api_url, is_healthy=is_healthy, saved_areas=saved_areas)


if __name__ == "__main__":
    main()
