import cv2
import sys

cam_num = 0
if sys.argv[1]:
    cam_num = int(sys.argv[1])
cap = cv2.VideoCapture(cam_num)

while True:
    ret, frame = cap.read()	
    if not ret:
        break
	
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	
cap.release()
cv2.destroyAllWindows()

