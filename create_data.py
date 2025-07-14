import cv2
import os

# Path to the Haar cascade file for face detection
haar_file = 'haarcascade_frontalface_default.xml'

# Directory path to store the face dataset
datasets_path = r"D:\Face\datasets"

# Name of the sub-directory (person's name) for the dataset
sub_data = 'mandar'

# Path to the directory where this person's images will be stored
person_dir = os.path.join(datasets_path, sub_data)

# Create the directory if it doesn't exist
if not os.path.isdir(person_dir):
    os.makedirs(person_dir)

# Dimensions for resizing the captured face images
width, height = 130, 100

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Start capturing video from the webcam (0 is usually the default webcam)
webcam = cv2.VideoCapture(0)

# The program loops until it has captured 30 images of the face
count = 1
while count <= 100:
    # Read a frame from the webcam
    _, frame = webcam.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region from the grayscale frame
        face = gray[y:y + h, x:x + w]

        # Resize the face region to the desired dimensions
        face_resize = cv2.resize(face, (width, height))

        # Save the resized face image to the person's directory
        img_path = os.path.join(person_dir, f'{count}.png')
        cv2.imwrite(img_path, face_resize)

        # Increment the count of captured images
        count += 1

    # Display the frame with detected faces
    cv2.imshow('OpenCV - Face Dataset Collection', frame)

    # Check for the 'Esc' key press to stop capturing
    key = cv2.waitKey(10)
    if key == 27:  # Esc key
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()

print(f" zala !!!'{sub_data}'. {count - 1} phuto kadle.")
