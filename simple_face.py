import cv2
import face_recognition
import numpy as np

img = cv2.imread("pavan10.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encodings = face_recognition.face_encodings(rgb_img)

img2 = cv2.imread("images/pavan.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encodings2 = face_recognition.face_encodings(rgb_img2)

result = face_recognition.compare_faces(img_encodings, img_encodings2[0])
print("Result:", result)

cv2.imshow("Img", img)
cv2.imshow("Img2", img2)
cv2.waitKey(0)
