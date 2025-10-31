import cv2
import numpy as np
from mediapipe.python.solutions import hands

from backend.models.base_model import BaseModelInference
from backend.settings.model_config import HandsConfig
from backend.utils.logging import logging_default


class MediapipeHandsModel(BaseModelInference):
    def __init__(self, model_settings : HandsConfig):
        super().__init__()

        # Load Model configurations first 
        self.load_configurations(model_settings)

        # Initiate the model
        self.load_model(None)

    def load_configurations(self, config : HandsConfig):
        """
        Load the detection of model settings configurations from a configuration JSON file.

        Parameters
        ----------
        path : config
        """

        logging_default.info("Loading hands detection configs and model configuration")

        self.min_detection_confidence = config.min_detection_confidence
        self.min_tracking_confidence = config.min_tracking_confidence

        # Log the configurations loaded
        logging_default.info(
            "Loaded configuration - "
            "Min Tracking Confidence: {min_tracking_confidence:.2f}, Min Detection Confidence: {min_detection_confidence:.2f}",
            min_tracking_confidence=self.min_tracking_confidence,
            min_detection_confidence=self.min_detection_confidence
        )
        return

    def load_model(self, model_path : str):
        self.hands_pose = hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def preprocess(self, image : np.ndarray):
        """
        This function is to preprocess the image before going to the Mediapipe model.
        Process an BGR image and return the image in RGB format

        Parameters
        ----------
        image : np.ndarray
            The image frame of which want to get the face landmark
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def inference(self, image: np.ndarray, preprocessed: bool = True):
        """
        Perform inference using the MediaPipe Hands model, extract the relevant hand landmarks,
        and return both the raw MediaPipe result and a list of hand landmarks.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        preprocessed : bool
            Whether the image is already in RGB format.

        Returns
        -------
        tuple
            (raw_mp_result, hand_landmarks)
            - raw_mp_result: The raw result from MediaPipe (for debugging or further processing).
            - hand_landmarks: List of lists of (x, y, z) tuples for each detected hand.
        """
        if not preprocessed:
            image = self.preprocess(image)
        raw_mp_result = self.hands_pose.process(image)

        hand_landmarks = []
        if raw_mp_result.multi_hand_landmarks:
            for hand_landmark in raw_mp_result.multi_hand_landmarks:
                hand_landmarks.append([
                    (lm.x, lm.y, lm.z) for lm in hand_landmark.landmark
                ])

        return raw_mp_result, hand_landmarks