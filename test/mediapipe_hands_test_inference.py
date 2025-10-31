import cv2
import numpy as np
import sys
import logging
import mediapipe as mp
from mediapipe.python.solutions import hands

# ==============================================================================
# 1. IMPORT YOUR CUSTOM LOGGER
# ==============================================================================
# This assumes 'logging.py' is in the same directory or accessible in your PYTHONPATH
try:
    from logging import logging_default
except ImportError:
    print("Error: Could not import 'logging_default' from logging.py.")
    print("Please make sure 'logging.py' is in the same directory.")
    sys.exit(1)

# ==============================================================================
# 2. MOCK CLASSES (Placeholders for your imported code)
# ==============================================================================
class BaseModelInference(object):
    """
    Mock base class since it wasn't provided.
    """
    def __init__(self):
        logging_default.debug("BaseModelInference initialized")

class HandsConfig:
    """
    Mock config class to make the model runnable.
    """
    def __init__(self, min_det=0.5, min_track=0.5):
        self.min_detection_confidence = min_det
        self.min_tracking_confidence = min_track
        logging_default.debug("HandsConfig initialized")


# ==============================================================================
# 3. YOUR HANDS MODEL CLASS
# ==============================================================================
class MediapipeHandsModel(BaseModelInference):
    def __init__(self, model_settings : HandsConfig):
        super().__init__()
        self.load_configurations(model_settings)
        self.load_model(None)

    def load_configurations(self, config : HandsConfig):
        logging_default.info("Loading hands detection configs...")
        self.min_detection_confidence = config.min_detection_confidence
        self.min_tracking_confidence = config.min_tracking_confidence
        logging_default.info(
            f"Loaded configuration - Min Detection: {self.min_detection_confidence}, "
            f"Min Tracking: {self.min_tracking_confidence}"
        )
        return

    def load_model(self, model_path : str):
        self.hands_pose = hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        logging_default.info("MediaPipe Hands model loaded.")

    def preprocess(self, image : np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def inference(self, image : np.ndarray, preprocessed : bool = True):
        """
        Perform inference and return both the raw MediaPipe result 
        and the custom (x, y, z) coordinate list.
        """
        if not preprocessed:
            image = self.preprocess(image)
        
        # Performance optimization
        image.flags.writeable = False
        inference_result = self.hands_pose.process(image)
        # Make image writeable again for drawing
        image.flags.writeable = True
        
        hand_landmarks_list = []
        if inference_result.multi_hand_landmarks:
            for hand_landmark in inference_result.multi_hand_landmarks:
                hand_landmarks_list.append([
                    (lm.x, lm.y, lm.z) for lm in hand_landmark.landmark
                ])

        # Return both the raw result (for drawing) and the xyz list (for your data)
        return inference_result, hand_landmarks_list

# ==============================================================================
# 4. TEST SCRIPT MAIN FUNCTION
# ==============================================================================
def run_hand_tracking_test():
    logging_default.info("Starting hand tracking test program.")
    
    # 1. Initialize Config and Model
    config = HandsConfig(min_det=0.7, min_track=0.7)
    hand_tracker = MediapipeHandsModel(model_settings=config)
    
    # 2. Initialize MediaPipe Drawing Utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # 3. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging_default.error("Cannot open webcam!")
        return
    
    logging_default.info("Webcam opened. Starting main loop... (Press 'ESC' to exit)")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            logging_default.warning("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a natural, selfie-view display
        image = cv2.flip(image, 1)

        # 4. Perform Inference
        raw_mp_result, xyz_coordinates_list = hand_tracker.inference(image, preprocessed=False)

        # 5. *** READ AND LOG THE 3D COORDINATES ***
        if xyz_coordinates_list:
            # `xyz_coordinates_list` is a list of hands.
            # Let's get the first hand:
            first_hand = xyz_coordinates_list[0]
            
            # `first_hand` is a list of 21 landmarks.
            # Let's get the WRIST (landmark 0):
            wrist_coords = first_hand[0] # This is your (x, y, z) tuple
            
            # Log the coordinates using your logger
            logging_default.info(
                f"Detected {len(xyz_coordinates_list)} hand(s). "
                f"Hand 0 Wrist (x,y,z): "
                f"({wrist_coords[0]:.3f}, {wrist_coords[1]:.3f}, {wrist_coords[2]:.3f})"
            )
        
        # 6. Draw the hand annotations on the image for visual feedback
        if raw_mp_result.multi_hand_landmarks:
            for hand_landmarks in raw_mp_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # 7. Display the image
        cv2.imshow('Mediapipe Hands Test', image)

        # 8. Exit condition
        if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' key
            logging_default.info("Escape key pressed. Exiting...")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    logging_default.info("Program finished.")


# ==============================================================================
# 5. RUN THE TEST
# ==============================================================================
if __name__ == "__main__":
    run_hand_tracking_test()