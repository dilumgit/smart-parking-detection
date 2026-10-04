"""
Smart Parking AI - Parking Configuration and Area Management Module.

This module provides:
- ParkingConfig: Manages single parking space configuration files.
- ParkingAreaManager: Manages multiple named parking area configurations with file-based persistence.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ParkingConfig:
    """
    Manages parking space polygon configurations.
    Loads and validates parking space boundaries from JSON configuration files.
    """

    def __init__(self, config_path: Union[str, Path] = "data/parking_configs/parking.json") -> None:
        """
        Initialize the ParkingConfig loader.

        Args:
            config_path: Path to the JSON parking configuration file.
        """
        self.config_path = Path(config_path)
        self.data: Dict[str, Any] = {"image": "", "spaces": []}
        self._is_loaded = False

    def load(self) -> Dict[str, Any]:
        """
        Load and validate the parking configuration from JSON.

        Returns:
            Dict[str, Any]: Configuration dictionary with 'image' and 'spaces'.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration structure or polygon data is invalid.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON format in {self.config_path}: {exc}") from exc

        self._validate_schema(content)
        self.data = content
        self._is_loaded = True
        return self.data

    def get_spaces(self) -> List[Dict[str, Any]]:
        """
        Get the list of parking spaces. Automatically loads config if not already loaded.

        Returns:
            List[Dict[str, Any]]: List of parking space dictionaries with 'id' and 'polygon'.
        """
        if not self._is_loaded:
            self.load()
        spaces = self.data.get("spaces", [])
        # Ensure standard 'id' field is present even if 'space_id' was used
        normalized = []
        for s in spaces:
            space_id = s.get("space_id") or s.get("id")
            normalized.append({
                "id": space_id,
                "space_id": space_id,
                "polygon": s["polygon"],
            })
        return normalized

    def get_image_path(self) -> str:
        """
        Get the image path associated with this parking configuration.

        Returns:
            str: Relative or absolute image path string.
        """
        if not self._is_loaded:
            self.load()
        return self.data.get("reference_image") or self.data.get("image", "")

    def save(self, image_path: str, spaces: List[Dict[str, Any]]) -> None:
        """
        Save parking configuration to JSON file.

        Args:
            image_path: Associated image path.
            spaces: List of parking spaces with 'id' and 'polygon'.
        """
        config_data = {
            "image": str(image_path),
            "spaces": spaces,
        }
        self._validate_schema(config_data)

        # Ensure parent directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        self.data = config_data
        self._is_loaded = True

    @staticmethod
    def _validate_schema(content: Any) -> None:
        """
        Validate that content matches the required parking configuration schema.
        """
        if not isinstance(content, dict):
            raise ValueError("Configuration root must be a JSON object (dict).")

        if "spaces" not in content or not isinstance(content["spaces"], list):
            raise ValueError("Configuration must contain a 'spaces' list.")

        for idx, space in enumerate(content["spaces"]):
            if not isinstance(space, dict):
                raise ValueError(f"Space at index {idx} must be an object.")

            space_id = space.get("space_id") or space.get("id")
            if not space_id or not isinstance(space_id, str) or not space_id.strip():
                raise ValueError(f"Space at index {idx} must have a non-empty string 'id' or 'space_id'.")

            if "polygon" not in space or not isinstance(space["polygon"], list):
                raise ValueError(f"Space '{space_id}' must have a 'polygon' list.")

            polygon = space["polygon"]
            if len(polygon) != 4:
                raise ValueError(
                    f"Space '{space_id}' polygon must contain exactly 4 points, got {len(polygon)}."
                )

            for pt_idx, pt in enumerate(polygon):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    raise ValueError(
                        f"Space '{space_id}' point {pt_idx} must be a 2-element coordinate [x, y]."
                    )
                x, y = pt
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    raise ValueError(
                        f"Space '{space_id}' coordinates must be numbers, got [{type(x)}, {type(y)}]."
                    )


class ParkingAreaManager:
    """
    Manages multiple saved parking area configurations stored as individual JSON files.
    """

    def __init__(self, configs_dir: Union[str, Path] = "data/parking_configs") -> None:
        """
        Initialize the ParkingAreaManager.

        Args:
            configs_dir: Directory path where parking area JSON configurations are stored.
        """
        self.configs_dir = Path(configs_dir)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def _get_area_filepath(self, parking_area_id: str) -> Path:
        """Get the file path for a specific parking area ID."""
        # Sanitize area ID to prevent directory traversal
        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", parking_area_id.strip())
        return self.configs_dir / f"{clean_id}.json"

    def validate_area(self, area_data: Any) -> None:
        """
        Validate that the parking area data meets all schema and integrity requirements.

        Args:
            area_data: Dictionary containing parking area definition.

        Raises:
            ValueError: If any field is missing or invalid.
        """
        if not isinstance(area_data, dict):
            raise ValueError("Parking area data must be a JSON object (dict).")

        # 1. Validate parking_area_id
        area_id = area_data.get("parking_area_id")
        if not area_id or not isinstance(area_id, str) or not area_id.strip():
            raise ValueError("Parking area must have a non-empty string 'parking_area_id'.")
        if not re.match(r"^[a-zA-Z0-9_-]+$", area_id.strip()):
            raise ValueError(
                "Field 'parking_area_id' may only contain alphanumeric characters, hyphens, and underscores."
            )

        # 2. Validate name
        name = area_data.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Parking area must have a non-empty string 'name'.")

        # 3. Validate spaces
        spaces = area_data.get("spaces")
        if not isinstance(spaces, list) or len(spaces) == 0:
            raise ValueError("Parking area must contain at least one parking space in 'spaces'.")

        seen_space_ids = set()
        for idx, space in enumerate(spaces):
            if not isinstance(space, dict):
                raise ValueError(f"Space at index {idx} must be an object.")

            space_id = space.get("space_id") or space.get("id")
            if not space_id or not isinstance(space_id, str) or not space_id.strip():
                raise ValueError(f"Space at index {idx} must have a non-empty 'space_id'.")

            clean_space_id = space_id.strip()
            if clean_space_id in seen_space_ids:
                raise ValueError(f"Duplicate space ID '{clean_space_id}' found in parking area.")
            seen_space_ids.add(clean_space_id)

            polygon = space.get("polygon")
            if not isinstance(polygon, list) or len(polygon) != 4:
                raise ValueError(f"Space '{clean_space_id}' polygon must contain exactly 4 points.")

            for pt_idx, pt in enumerate(polygon):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    raise ValueError(f"Space '{clean_space_id}' point {pt_idx} must be [x, y].")
                x, y = pt
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    raise ValueError(f"Space '{clean_space_id}' coordinates must be numeric.")

    def create_area(self, area_data: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        """
        Create and persist a new parking area configuration.

        Args:
            area_data: Dictionary containing parking area data.
            overwrite: If False, raises ValueError if the area ID already exists.

        Returns:
            Dict[str, Any]: Saved parking area data.
        """
        self.validate_area(area_data)

        area_id = area_data["parking_area_id"].strip()
        filepath = self._get_area_filepath(area_id)

        if filepath.exists() and not overwrite:
            raise ValueError(f"Parking area with ID '{area_id}' already exists.")

        # Normalize spaces format
        normalized_spaces = []
        for s in area_data["spaces"]:
            s_id = (s.get("space_id") or s.get("id", "")).strip()
            normalized_spaces.append({
                "space_id": s_id,
                "polygon": s["polygon"],
            })

        clean_record = {
            "parking_area_id": area_id,
            "name": area_data["name"].strip(),
            "reference_image": str(area_data.get("reference_image") or area_data.get("image") or "1.jpg"),
            "spaces": normalized_spaces,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(clean_record, f, indent=2)

        return clean_record

    def get_area(self, parking_area_id: str) -> Dict[str, Any]:
        """
        Load a saved parking area by its ID.

        Args:
            parking_area_id: Identifier of the parking area.

        Returns:
            Dict[str, Any]: Loaded parking area data.

        Raises:
            FileNotFoundError: If the parking area does not exist.
        """
        filepath = self._get_area_filepath(parking_area_id)
        if not filepath.exists():
            # Check legacy parking.json fallback if requested
            if parking_area_id.lower() in ("default", "parking"):
                legacy_file = self.configs_dir / "parking.json"
                if legacy_file.exists():
                    with open(legacy_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return {
                        "parking_area_id": "parking_area_001",
                        "name": "Default Parking Area",
                        "reference_image": data.get("image", "1.jpg"),
                        "spaces": [
                            {"space_id": s.get("id") or s.get("space_id"), "polygon": s["polygon"]}
                            for s in data.get("spaces", [])
                        ],
                    }
            raise FileNotFoundError(f"Parking area '{parking_area_id}' not found.")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt JSON in parking area file '{filepath}': {exc}") from exc

        return data

    def list_areas(self) -> List[Dict[str, Any]]:
        """
        List all saved parking area configurations.

        Returns:
            List[Dict[str, Any]]: List of area summary records.
        """
        areas: List[Dict[str, Any]] = []

        if not self.configs_dir.exists():
            return areas

        for file in sorted(self.configs_dir.glob("*.json")):
            # Skip the legacy single-config parking.json unless it has parking_area_id
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = json.load(f)
            except Exception:
                continue

            if not isinstance(content, dict):
                continue

            # Check if this is a named parking area or legacy config
            if "parking_area_id" in content:
                area_id = content["parking_area_id"]
                name = content.get("name", area_id)
                ref_img = content.get("reference_image") or content.get("image", "1.jpg")
                spaces = content.get("spaces", [])
            elif file.name == "parking.json":
                area_id = "default"
                name = "Default Area"
                ref_img = content.get("image", "1.jpg")
                spaces = content.get("spaces", [])
            else:
                continue

            areas.append({
                "parking_area_id": area_id,
                "name": name,
                "reference_image": ref_img,
                "total_spaces": len(spaces) if isinstance(spaces, list) else 0,
            })

        return areas
