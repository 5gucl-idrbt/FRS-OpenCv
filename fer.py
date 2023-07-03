from fer import FER
# import FER as fer
import cv2

test_img = cv2.imread("images/sun.jpg")
analysis = emotion_detector.detect_emotions(test_img)

print(analysis)