import cv2
import numpy as np
import os

# Constants
DATASETS_DIR = 'datasets'
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
FACE_WIDTH, FACE_HEIGHT = 130, 100

def load_face_dataset():
    # Create lists to store images and corresponding labels (IDs)
    images = []
    labels = []
    names = {}
    label_id = 0

    # Traverse through each subdirectory in the datasets directory
    for subdir_name in os.listdir(DATASETS_DIR):
        subdir_path = os.path.join(DATASETS_DIR, subdir_name)
        
        if os.path.isdir(subdir_path):
            # Assign a unique label ID to each subdirectory (person)
            names[label_id] = subdir_name
            
            # Load each image in the subdirectory
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                
                # Read image in grayscale and append to images list
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, (FACE_WIDTH, FACE_HEIGHT)))
                    labels.append(label_id)
            
            label_id += 1
    
    # Convert lists to numpy arrays
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    
    return images, labels, names

def main():
    print('lighta khali bss!!!')
    
    # Load the pre-trained LBPH (Local Binary Patterns Histograms) Face Recognizer model
    model = cv2.face.LBPHFaceRecognizer_create()
    
    # Load the face dataset
    images, labels, names = load_face_dataset()
    
    # Train the face recognizer model using the loaded dataset
    model.train(images, labels)
    
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_CASCADE_FILE)
    
    # Open webcam (index 0) for capturing video stream
    webcam = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the webcam
        ret, frame = webcam.read()
        
        if not ret:
            print("Failed to capture frame from webcam.")
            break
        
        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract face region and resize for prediction
            face_roi = gray_frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (FACE_WIDTH, FACE_HEIGHT))
            
            # Predict the label (ID) of the detected face
            label_id, confidence = model.predict(face_resized)
            
            # Determine the name of the predicted person
            if confidence < 500:
                person_name = names.get(label_id, 'Unknown')
                text = f'{person_name} - confidence : {int(100 - confidence)}%'
            else:
                text = 'Unknown'
            
            # Display the recognized name or 'Unknown' on the frame
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the frame with detected faces
        cv2.imshow('Face Recogination!!!', frame)
        
        # Check for 'Esc' key press to exit
        if cv2.waitKey(1) == 27:
            break
    
    # Release the webcam and close all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
