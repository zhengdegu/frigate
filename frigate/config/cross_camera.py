"""Cross-camera tracking configuration."""

from typing import Optional

from pydantic import Field

from frigate.config.base import FrigateBaseModel


class CameraLink(FrigateBaseModel):
    """Defines the spatial relationship between two cameras."""

    source: str = Field(title="Source camera name.")
    target: str = Field(title="Target camera name.")
    min_seconds: float = Field(default=3.0, title="Minimum transfer time in seconds.")
    max_seconds: float = Field(default=60.0, title="Maximum transfer time in seconds.")
    bidirectional: bool = Field(default=True, title="Whether the link works both ways.")


class CrossCameraConfig(FrigateBaseModel):
    """Cross-camera tracking configuration."""

    enabled: bool = Field(default=False, title="Enable cross-camera tracking.")
    tracked_objects: list[str] = Field(
        default=["car"], title="Object labels to track across cameras."
    )
    plate_match_threshold: float = Field(
        default=0.8, title="Fuzzy match threshold for license plates (0-1)."
    )
    color_weight: float = Field(
        default=0.4, title="Weight for color similarity in combined score."
    )
    type_weight: float = Field(
        default=0.3, title="Weight for vehicle type similarity in combined score."
    )
    appearance_weight: float = Field(
        default=0.3, title="Weight for histogram appearance similarity."
    )
    match_threshold: float = Field(
        default=0.55, title="Minimum combined score for a match without plate."
    )
    gallery_max_age: float = Field(
        default=300.0, title="Seconds to keep a disappeared track in the gallery."
    )
    topology: list[CameraLink] = Field(
        default=[], title="Camera spatial relationships."
    )
