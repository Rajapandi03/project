import cv2
import numpy as np
import os
import sys

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  # Path to the dataset directory

print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

# Walk through the dataset directory to collect images and labels
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        print(f"Processing folder: {subdir}")  # Debugging statement
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        print(f"Subject path: {subjectpath}")  # Debugging statement
        if not os.path.exists(subjectpath):
            print(f"Path does not exist: {subjectpath}")  # Debugging statement
            continue
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            print(f"Loading image: {path}")  # Debugging statement
            if not os.path.isfile(path):
                print(f"File does not exist: {path}")  # Debugging statement
                continue
            img = cv2.imread(path, 0)
            if img is None:
                print(f"Failed to load image: {path}")  # Debugging statement
            else:
                images.append(img)
                labels.append(int(id))
        id += 1

# Check the number of loaded images
print(f"Number of images loaded: {len(images)}")
if len(images) == 0 or len(labels) == 0:
    print("No images found in the dataset. Please check the dataset path and ensure images are available.")
    sys.exit()

# Convert the list of images and labels to numpy arrays
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Set the image size for resizing
(width, height) = (130, 100)

# Create and train the model
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

# Load the Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    raise IOError('Haar Cascade file not found')

try:
    # Initialize webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise IOError('Cannot open webcam')

    cnt = 0
    while True:
        ret, im = webcam.read()
        if not ret:
            print("Failed to grab frame")
            break

        if im is None or im.size == 0:
            print("Empty frame captured")
            continue

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            prediction = model.predict(face_resize)

            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if prediction[1] < 800:
                cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
                print(names[prediction[0]])
                cnt = 0
            else:
                cnt += 1
                cv2.putText(im, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                if cnt > 100:
                    print("Unknown Person")
                    cv2.imwrite("input.jpg", im)
                    cnt = 0

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:  # ESC key to exit
            break

finally:
    webcam.release()
    cv2.destroyAllWindows()
