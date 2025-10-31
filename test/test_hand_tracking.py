import cv2
import numpy as np
import sys
import logging
import mediapipe as mp
from mediapipe.python.solutions import hands

# ==============================================================================
# 1. IMPORT YOUR CUSTOM MODULES
# ==============================================================================
# This assumes your script is run from a directory where 'backend' is visible
try:
    # Import the logger
    from backend.utils.logging import logging_default
    
    # Import the model class you saved
    from backend.models.mediapipe.mediapipe_hands_model import MediapipeHandsModel
    
    # Import the real config class (inferred from your original file's dependencies)
    from backend.settings.model_config import HandsConfig 
    
except ImportError as e:
    logging.critical(f"Error: Could not import necessary modules. {e}")
    logging.critical("Please make sure you are running this script from the root project")
    logging.critical("directory (the one containing the 'backend' folder).")
    sys.exit(1)

# === SECTIONS 2 AND 3 (Mock Classes and Model Definition) ARE NOW REMOVED ===


# ==============================================================================
# 4. TEST SCRIPT MAIN FUNCTION (Unchanged)
# ==============================================================================
def run_hand_tracking_test():
    logging_default.info("Starting hand tracking test program.")
    
    # 1. Initialize Config and Model
    # These now use the *real* classes you imported
    config = HandsConfig(min_detection_confidence=0.7, min_tracking_confidence=0.7) 
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
        # This calls the .inference() method from your imported model class
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
# 5. RUN THE TEST (Unchanged)
# ==============================================================================
if __name__ == "__main__":
    run_hand_tracking_test()