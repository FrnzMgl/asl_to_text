from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, JsonResponse
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import tempfile
# Create your views here.
def pages(request):
    return HttpResponse("This is the pages view")

def home(request):
    context = {}
    return render(request, 'home/index.html', context)


def lstm(request):
    context = {}
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
    video_file = request.FILES['video']
    video_path = video_file.temporary_file_path()
    lexicon_path = os.path.join(settings.BASE_DIR, 'resources', 'lexicon.csv')
    lexicon_df = pd.read_csv(lexicon_path)
    lexicon_dict = dict(zip(lexicon_df['ASL Structure'], lexicon_df['Tagalog']))
    
    def match_translation(asl_structure):
        if asl_structure in lexicon_dict:
            return lexicon_dict[asl_structure]  # Return the English translation
        return "No match found"  # Default if no match exists

    model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'resources', 'asl_model13.h5'), compile=False)
    label_encoder_classes = np.load(os.path.join(settings.BASE_DIR, 'resources', 'classes.npy'))
    # MediaPipe setup for pose and hands
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    # Constants
    num_hand_keypoints = 21 * 2  # 21 landmarks * (x, y)
    num_pose_keypoints = 33 * 3  # 33 landmarks * (x, y, z)
    max_frames = 24  # Fixed number of frames for input
    confidence_threshold = 0.95  # Minimum confidence required for predictions

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
                    frame_data[i * 42:i * 42 +
                               len(hand_keypoints)] = hand_keypoints

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
                # Shape (1, max_frames, features)
                input_data = np.expand_dims(np.array(frames), axis=0)
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)

                if confidence >= confidence_threshold:
                    sign_name = label_encoder_classes[predicted_class]
                    predictions.append(sign_name)
                    sign_detected = True  # Set flag to True as a sign is detected

                    # Print the intermediate prediction to the console
                    print(
                        f"Predicted Sign: {sign_name} (Confidence: {confidence:.2f})")
                    
                    if not context.get('prediction'):
                        context['prediction'] = []
                        
                    context['prediction'].append({ 'sign': sign_name, 'confidence': float(confidence) })

                else:
                    # If confidence is below the threshold, print a message to the console
                    print(
                        f"Confidence below {confidence_threshold:.2f} for the frame.")

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


    # Test with a video
    translations = "".join(translate_signs_from_video(video_path))
    final_lexicon = match_translation(translations)
    print(final_lexicon)
# Store the final lexicon in the context
    context['final_translation'] = final_lexicon

    return JsonResponse({'message': 'Success', 'data': context}, status=200)

def linktrans(request):
    context = {}
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)

    video_url = request.POST.get("video_url")  # Get the YouTube URL
    if not video_url:
        return JsonResponse({'error': 'No video URL provided'}, status=400)

    try:
        video_path = download_youtube_video(video_url)  # Download YouTube video
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

    # Load Lexicon
    lexicon_path = os.path.join(settings.BASE_DIR, 'resources', 'lexicon.csv')
    lexicon_df = pd.read_csv(lexicon_path)
    lexicon_dict = dict(zip(lexicon_df['ASL Structure'], lexicon_df['Tagalog']))

    # Load Model
    model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'resources', 'asl_model13.h5'), compile=False)
    label_encoder_classes = np.load(os.path.join(settings.BASE_DIR, 'resources', 'classes.npy'))

    # Process Video and Get Translation
    final_translation = translate_signs_from_video(video_path, model, label_encoder_classes, lexicon_dict, context)

    context['final_translation'] = final_translation

    return JsonResponse({'message': 'Success', 'data': context}, status=200)

    

def v2t(request):
    return render(request, 'video/index.html')

def l2t(request):
    return render(request, 'link/index.html')

def test(request): 
    return render(request, 'test/index.html')



from django.http import StreamingHttpResponse
import cv2

# Function to generate video frames for streaming
def generate_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Endpoint to serve video frames for live streaming
def video_feed(request):
    video_path = request.GET.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        return HttpResponse("Invalid video path", status=400)

    return StreamingHttpResponse(generate_video_frames(video_path),
                                 content_type="multipart/x-mixed-replace; boundary=frame")

# Update your LSTM processing function to return the video path
def lstm(request):
    context = {}
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
    video_file = request.FILES['video']
    
    # Save the uploaded video to a temporary directory
    temp_video_path = os.path.join(settings.MEDIA_ROOT, video_file.name)
    with open(temp_video_path, 'wb+') as destination:
        for chunk in video_file.chunks():
            destination.write(chunk)

    # Process video using your model (Modify this if needed)
    translations = "example_translation"  # Replace with actual translation logic
    final_lexicon = "example_translation_in_tagalog"  # Replace with actual lexicon matching

    context['final_translation'] = final_lexicon

    return JsonResponse({'message': 'Success', 'data': context, 'video_path': temp_video_path}, status=200)