import cv2
import pyshine as ps
from simple_facerec import SimpleFacerec
from deepface import DeepFace

############## ADDED FACE EXPRESSION #############

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images_test/")

# Initialize WebRTC streamer with the WebRTC URL
streamer = ps.Streamer()

# Set WebRTC parameters
streamer.webrtc = True
streamer.webrtc_stream_id = 'stream1'

while True:
    # Capture video frame from the camera
    ret, frame = cv2.VideoCapture(0).read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Crop face region for emotional detection
        face_image = frame[y1:y2, x1:x2]

        try:
            # Perform emotional analysis
            emotions = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)

            # Get the dominant emotion label
            emotion_scores = emotions[0]['emotion']
            emotion_label = max(emotion_scores, key=emotion_scores.get)

            # Display the predicted emotion
            cv2.putText(frame, emotion_label, (x1, y2 + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 2)

        except ValueError:
            # Face detection failed
            print("Face could not be detected")

    # Stream the frame with overlays using WebRTC
    streamer.update_frame(frame)
    streamer.send_frame()

    key = cv2.waitKey(1)
    if key == 27:
        break

streamer.stop()
cv2.destroyAllWindows()
