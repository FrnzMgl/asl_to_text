import os
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from .utils import translate_signs_from_video  # Import your function to process video
from django.core.files.storage import FileSystemStorage

def process_video(request):
    if request.method == 'POST' and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']

        # Save the file temporarily on the server
        fs = FileSystemStorage(location=os.path.join(settings.BASE_DIR, 'uploaded_videos'))
        filename = fs.save(video_file.name, video_file)
        video_path = fs.path(filename)

        # Call the translation function to process the uploaded video
        translations = translate_signs_from_video(video_path)
        translated_signs = "".join(translations)

        # Clean up the uploaded file if needed
        os.remove(video_path)  # Optional: Delete the file after processing

        # Return the translated signs to be displayed on the same page
        return render(request, 'upload.html', {'translated_signs': translated_signs})

    return render(request, 'upload.html')
