import cv2

face_cascade = cv2.CascadeClassifier("./Hasa/haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier("./Hasa/haarcascade_eye.xml")
cap = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_COMPLEX
fontscale = 1
fontcolor = (203, 23, 252)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        cv2.putText(img, "Name: Viet", (x, y + h + 30), fontface, fontscale, fontcolor, 2)
        cv2.putText(img, "Age:20", (x, y + h + 60), fontface, fontscale, fontcolor, 2)
        cv2.putText(img, "Gender:Male", (x, y + h + 90), fontface, fontscale, fontcolor, 2)
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('Face Recogation', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
