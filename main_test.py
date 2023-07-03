import cv2
from simple_facerec import SimpleFacerec
from deepface import DeepFace

############## ADDED FACE EXPRESSION #############

# RTSP
rtsp_url = 'rtsp://192.168.138.116:8080/h264_ulaw.sdp'
rtmp_url = 'rtmp://192.168.138.121:1935/live/demo'

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images_test/")

# Load camera
# cap = cv2.VideoCapture(rtmp_url)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

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

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
