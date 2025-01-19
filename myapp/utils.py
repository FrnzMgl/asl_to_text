# utils.py (or the same file where translate_signs_from_video resides)

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd

# Your lexicon loading and translation logic
import os
from django.conf import settings

# Construct the correct path to your lexicon.csv file
model_path = os.path.join(settings.BASE_DIR, 'myapp', 'models', 'asl_model_letter.h5')
class_path = os.path.join(settings.BASE_DIR, 'myapp', 'models', 'classes.npy')

lexicon_path = os.path.join(settings.BASE_DIR, 'myapp', 'lexicon', 'lexicon.csv')
lexicon_df = pd.read_csv(lexicon_path)
lexicon_dict = dict(zip(lexicon_df['ASL Structure'], lexicon_df['English']))

def match_translation(asl_structure):
    if asl_structure in lexicon_dict:
        return lexicon_dict[asl_structure]  # Return the English translation
    return "No match found"  # Default if no match exists

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load label encoder classes
label_encoder_classes = np.load(class_path)
print("Loaded classes:", label_encoder_classes)

# MediaPipe setup for pose and hands
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Constants
num_hand_keypoints = 21 * 2  # 21 landmarks * (x, y)
num_pose_keypoints = 33 * 3  # 33 landmarks * (x, y, z)
max_frames = 24  # Fixed number of frames for input
confidence_threshold = 0.90  # Minimum confidence required for predictions

# List of pronouns for reordering
pronouns = {"me", "I", "you", "we", "he", "she", "they"}

# Function to extract keypoints from a frame
def extract_keypoints(frame):
    frame_data = np.zeros(num_hand_keypoints + num_pose_keypoints)

    # Process hands
    hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            if i < 2:  # Limit to two hands
                hand_keypoints = []
                for landmark in hand_landmarks.landmark:
                    hand_keypoints.extend([landmark.x, landmark.y])
                frame_data[i * 42:i * 42 + len(hand_keypoints)] = hand_keypoints

    # Process pose
    pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if pose_results.pose_landmarks:
        pose_keypoints = []
        for landmark in pose_results.pose_landmarks.landmark:
            pose_keypoints.extend([landmark.x, landmark.y, landmark.z])
        frame_data[num_hand_keypoints:] = pose_keypoints

    return frame_data

# Function to reorder and clean predictions
def reorder_translation(predictions):
    filtered = []
    pronoun = None
    action_word = None

    # Loop through predictions to find pronoun and action word
    for word in predictions:
        if word in pronouns and pronoun is None:  # Set the first pronoun
            pronoun = word
        elif word not in pronouns and action_word is None:  # Set the first action word
            action_word = word

    # Construct the final sentence
    if pronoun and action_word:
        filtered.append(f"{pronoun} {action_word}")
    return filtered

# Process video and translate multiple signs
def translate_signs_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    predictions = []
    sign_detected = False  # Flag to track if any valid sign was detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract keypoints from the frame
        keypoints = extract_keypoints(frame)
        frames.append(keypoints)

        # Once we have enough frames, predict the sign
        if len(frames) == max_frames:
            input_data = np.expand_dims(np.array(frames), axis=0)  # Shape (1, max_frames, features)
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)

            if confidence >= confidence_threshold:
                sign_name = label_encoder_classes[predicted_class]
                predictions.append(sign_name)
                sign_detected = True  # Set flag to True as a sign is detected

                # Print the intermediate prediction to the console
                print(f"Predicted Sign: {sign_name} (Confidence: {confidence:.2f})")

            else:
                # If confidence is below the threshold, print a message to the console
                print(f"Confidence below {confidence_threshold:.2f} for the frame.")

            frames = []  # Reset frames for the next sequence

    cap.release()

    # Combine consecutive identical predictions to remove duplicates
    final_predictions = []
    for i, pred in enumerate(predictions):
        if i == 0 or pred != predictions[i - 1]:
            final_predictions.append(pred)

    # Check if no sign was detected
    if not sign_detected:
        return ["No sign detected"]

    # Reorder and clean the final translations
    cleaned_translation = reorder_translation(final_predictions)
    return cleaned_translation
