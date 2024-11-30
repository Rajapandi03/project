import cv2
import numpy
import os
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(width, height) = (130, 100)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
cnt = 0

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 800:
            name = names[prediction[0]]
            cv2.putText(im, f'{name} - {prediction[1]:.0f}', (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
            print(name)
            engine.say(f"{name}")
            engine.runAndWait()
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person")
                engine.say("Unknown Person Detected")
                engine.runAndWait()
                cv2.imwrite("input.jpg", im)
                cnt = 0

    cv2.imshow('camera', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
