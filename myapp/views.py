from django.shortcuts import render
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from collections import deque, Counter
import cv2
import json
import time
from django.http import JsonResponse

# Load the trained model
model = load_model('myapp/models/asl_model_letter.h5')

# Manually recreate the label encoder with A-Z alphabet
labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Define the lexicon (expand as needed)
lexicon = {
    "HPA": "HELLO",
    "THANK": "THANK",
    "THANK YOU": "THANK YOU",
    "ILY": "I LOVE YOU",
    "APPLE": "APPLE",
    "BOOK": "BOOK",
    "CAT": "CAT",
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Buffers for keypoints, predictions, and smoothing
sequence_length = 20
frame_buffer = deque(maxlen=sequence_length)
prediction_buffer = deque(maxlen=15)  
smoothing_window = deque(maxlen=5)  

# Cooldown variable
cooldown_time = 5  
cooldown_counter = 0

def extract_keypoints(hand_landmarks):
    """Extract hand keypoints."""
    keypoints = [landmark.x for landmark in hand_landmarks.landmark] + \
                [landmark.y for landmark in hand_landmarks.landmark] + \
                [landmark.z for landmark in hand_landmarks.landmark]
    return np.array(keypoints)

def translate_to_word(letters, lexicon):
    """Translate a sequence of letters into a word using the lexicon."""
    word = ''.join(letters).upper()
    return lexicon.get(word, word)

def process_frame(frame):
    global cooldown_counter
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    if cooldown_counter > 0:
        cooldown_counter -= 1
    
    translated_word = ''
    
    if results.multi_hand_landmarks and cooldown_counter == 0:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = extract_keypoints(hand_landmarks)
            frame_buffer.append(keypoints)
            
            if len(frame_buffer) == sequence_length:
                keypoints_sequence = np.array(frame_buffer).reshape(1, sequence_length, 21*3)
                prediction = model.predict(keypoints_sequence)
                class_idx = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)
                
                if confidence >= 0.7:
                    sign_label = label_encoder.classes_[class_idx]
                    smoothing_window.append(sign_label)
                    
                    if len(smoothing_window) == smoothing_window.maxlen:
                        most_common_sign, count = Counter(smoothing_window).most_common(1)[0]
                        
                        if count >= 3:  # min_consecutive_predictions
                            if len(prediction_buffer) == 0 or most_common_sign != prediction_buffer[-1]:
                                prediction_buffer.append(most_common_sign)
                                cooldown_counter = cooldown_time

                            translated_word = translate_to_word(prediction_buffer, lexicon)

    return translated_word

def home(request):
    # Render the initial page with an empty result
    return render(request, 'home/index.html', {'result': ''})

def process_video(request):
    cap = cv2.VideoCapture("myapp/vid/input_video.mp4")  # Use the video input
    result = ''
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        translated_word = process_frame(frame)
        
        if translated_word:
            result = translated_word  # Capture the latest translation
        
        # Calculate percentage completion
        percentage_complete = (frame_idx + 1) / frame_count * 100
        
        # Simulate delay for processing
        time.sleep(0.1)  # Adjust based on actual processing time
    
    cap.release()
    
    # Return the final translation and percentage as a JSON response
    return JsonResponse({'result': result, 'percentage': percentage_complete})
