import json

from pydantic import BaseModel

MODEL_CONFIG_PATH = "config/mediapipe_model_settings.json"

class FaceMeshConfig(BaseModel):
    static_image_mode: bool
    refine_landmarks: bool
    max_number_face_detection: int
    min_detection_confidence: float
    min_tracking_confidence: float

class PoseConfig(BaseModel):
    min_detection_confidence: float
    min_tracking_confidence: float
    enable_segmentation: bool
    smooth_segmentation: bool
    smooth_landmarks: bool
    static_image_mode: bool

class HandsConfig(BaseModel):
    min_detection_confidence: float
    min_tracking_confidence: float

class ModelConfig(BaseModel):
    face: FaceMeshConfig
    pose: PoseConfig
    hands: HandsConfig

    @classmethod
    def load(cls, path: str = MODEL_CONFIG_PATH):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
# Load the config once
model_settings = ModelConfig.load()