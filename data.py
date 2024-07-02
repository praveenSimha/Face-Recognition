import os
import cv2

# Define the Haar cascade file and the dataset path
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
sub_data = 'praveen'
path = os.path.join(datasets, sub_data)

# Create the dataset directory if it doesn't exist
os.makedirs(path, exist_ok=True)

# Define the width and height for the resized face images
width, height = 130, 100

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)

# Open a connection to the webcam (use 0 for default webcam)
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = 1
while count <= 30:
    # Capture frame from the webcam
    ret, im = webcam.read()
    if not ret or im is None:
        print("Error: Failed to capture frame.")
        continue

    # Convert the image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Extract the face area from the grayscale image
        face = gray[y:y + h, x:x + w]
        
        # Resize the face area to the defined width and height
        face_resize = cv2.resize(face, (width, height))
        
        # Save the face image to the dataset directory
        face_path = os.path.join(path, f'{count}.png')
        cv2.imwrite(face_path, face_resize)
        
        count += 1
        if count > 30:
            break
    
    # Display the captured frame
    cv2.imshow('OpenCV', im)
    
    # Break the loop if the ESC key is pressed
    if cv2.waitKey(10) == 27:
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
