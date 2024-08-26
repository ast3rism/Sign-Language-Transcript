# Testing the classifier
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import warnings

#ignore deprecated warnings
warnings.filterwarnings("ignore")

# Load trained model
model_dict = pickle.load(open(os.path.join('path/to/model', 'model.p'), 'rb'))
model = model_dict['model']

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

# Label mapping based on your trained labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
               13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'U', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

try:
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        # Check if the frame is read correctly
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        H, W, _ = frame.shape

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe
        try:
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Make a prediction
                prediction = model.predict([data_aux])

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                # Draw prediction results on the frame
                cv2.putText(frame, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Sign Language Transcript', frame)

            # Check for 'q' key press and close window if pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except:
            if len(results.multi_hand_landmarks) > 1:
                txt = "Cannot process multiple hands"
                cv2.putText(frame, txt, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.imshow('Sign Language Transcript', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            else:
                txt = "Unknown"
                cv2.putText(frame, txt, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.imshow('Sign Language Transcript', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break                

finally:
    cap.release()
    cv2.destroyAllWindows()