import pathlib
import cv2


path_cascade = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(path_cascade)

classifier = cv2.CascadeClassifier(str(path_cascade))

camera =cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        grayscale, 
        scaleFactor = 1.3, 
        minNeighbors = 5, 
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )


    for (x, y, width, heigth) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + heigth), (0, 255, 0), 2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()