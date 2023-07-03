# Used deepface conda env (Internal ref)

import cv2
from deepface import DeepFace

################### FACE EMOTION DETECTION USING DEEPFACE AND OTHER MODELS ##################

# Load the pre-trained face recognition model
DeepFace.stream("database", model_name="Facenet512") #VGG-Face is the default model

# Face models
# Use model_name = with different face models mentioned below
# VGG-Face
# Facenet
# OpenFace
# DeepFace

# models = [
#   "VGG-Face", 
#   "Facenet", 
#   "Facenet512", 
#   "OpenFace", 
#   "DeepFace", 
#   "DeepID", 
#   "ArcFace", 
#   "Dlib", 
#   "SFace",
# ]

# Model	      LFW Score	  YTF Score
# Facenet512	99.65%	    -
# SFace	        99.60%	    -
# ArcFace	    99.41%	    -
# Dlib	        99.38 %	    -
# Facenet	    99.20%	    -
# VGG-Face	    98.78%	   97.40%
# Human-beings  97.53%	    -
# OpenFace	    93.80%	    -
# DeepID	      -	       97.05%

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Perform face recognition and analysis on the frame
    result = DeepFace.analyze(frame, actions=['emotion'])

    # Get the dominant emotion
    emotions = result['emotion']
    emotion_label = max(emotions, key=emotions.get)

    # Display the dominant emotion on the frame
    cv2.putText(frame, emotion_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
