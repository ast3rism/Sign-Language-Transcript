# Creating Dataset
import os
import pickle
import mediapipe as mp
import cv2
import warnings

#ignore deprecated warnings
warnings.filterwarnings("ignore")

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
DATA_DIR = "path/to/training/image"

data = []
labels = []
problematic_img = []
c = 0

# Iterate over each directory representing a sign label
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    # Iterate over each image in the label directory
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        # Read image using OpenCV
        img = cv2.imread(img_path)

        # Convert BGR image to RGB for Mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Process the detected hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                for landmark in hand_landmarks.landmark:
                    # Normalize the coordinates
                    x = landmark.x / img.shape[1]
                    y = landmark.y / img.shape[0]
                    data_aux.append(x)
                    data_aux.append(y)
                data.append(data_aux)
                labels.append(label)

        if not results.multi_hand_landmarks:
            problematic_img.append(img_path)
        
        else:
            c += 1

hands.close()

with open("ProblematicImages.txt", "w") as f:
    for item in problematic_img:
        f.write("%s\n" % item)

# Save data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(c)