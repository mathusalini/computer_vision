![facefeatures](https://github.com/user-attachments/assets/7301a0f3-15c9-4573-b138-f7e3632488fa)

## Facial Feature Detection Using Haar Cascade

The diagram above visualizes how facial features such as eyes, nose, and mouth are detected using Haar-like features in a Haar Cascade algorithm. This process is essential for:

1. **Detecting the Position of a Face**: The Haar Cascade algorithm scans the image to locate the position of a face based on the identified features.
2. **Classifying Sleep Status**: Once the face is detected, the algorithm can analyze the state of the eyes to classify whether a person is asleep or awake.


- **Eyes**: The presence or absence of open eyes is a critical factor in determining if a person is sleeping.
- **Nose**: Helps in accurately positioning the face and recognizing facial orientation.
- **Mouth**: Also plays a role in understanding the overall facial expression.

##Face Detection Workflow
![flowchart](https://github.com/user-attachments/assets/db9b87c0-1c2f-4f1e-a84c-1c5fa540f877)

# Sleeping Detector Code Explanation

This document provides a detailed explanation of the sleeping detector code that uses OpenCV for face and eye detection.

## Code Explanation

import cv2
import numpy as np
Import Libraries: The code begins by importing the OpenCV library (cv2) for image processing and NumPy (np) for numerical operations.

# Load the pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
Load Haar Cascades: The pre-trained Haar cascade classifiers for face detection and eye detection are loaded. These classifiers are XML files that contain the trained model used for detecting faces and eyes in images.


# Function to detect eyes and classify sleep/awake based on eye detection
def detect_sleep(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Define Function: A function named detect_sleep is defined. This function will take a video frame as input and process it to detect faces and eyes.

Convert to Grayscale: The captured frame is converted to grayscale because face and eye detection is typically performed on grayscale images for efficiency.

    # Detect face in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
Face Detection: The detectMultiScale method is used to detect faces in the grayscale image. The parameters 1.3 (scale factor) and 5 (minimum neighbors) control the detection sensitivity. A higher scale factor and minimum neighbors will reduce false positives.


    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
Iterate Through Detected Faces: For each detected face, the coordinates (x, y) (top-left corner) and (w, h) (width and height) are retrieved.

Draw Rectangle: A blue rectangle is drawn around the detected face using cv2.rectangle.


        # Region of interest (face area) for eye detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
Define Region of Interest (ROI): The region of interest for eye detection is defined based on the detected face area. Both grayscale and color versions of the ROI are created.


        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
Eye Detection: The eye cascade classifier is used to detect eyes within the defined face region.


        if len(eyes) >= 2:
            # Eyes detected -> Awake
            cv2.putText(frame, "Awake", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Less than two eyes detected -> Sleeping
            cv2.putText(frame, "Sleeping", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
Classify Sleep/Awake: If two or more eyes are detected, the person is classified as "Awake," and green text is displayed. If less than two eyes are detected, the person is classified as "Sleeping," and red text is displayed.


        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
Draw Rectangles Around Eyes: For each detected eye, a green rectangle is drawn around it within the ROI.


    return frame
Return the Processed Frame: The modified frame (with rectangles and text) is returned.


# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
Initialize Webcam: The webcam is accessed using OpenCV's VideoCapture function. The parameter 0 indicates the default camera.


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
Start Frame Capture Loop: A loop is initiated to continuously capture frames from the webcam.

    # If frame not captured correctly, break the loop
    if not ret:
        break
Check Frame Capture: If the frame is not captured correctly, the loop breaks.


    # Call the sleep detection function
    frame = detect_sleep(frame)
Call the Detection Function: The detect_sleep function is called to process the captured frame.


    # Display the resulting frame
    cv2.imshow('Sleeping Detector', frame)
Display the Frame: The resulting frame with detected faces and classified sleep status is displayed in a window.

    # Press 'q' to exit the webcam window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
Exit Condition: The loop will exit if the 'q' key is pressed.

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
Cleanup: After exiting the loop, the webcam is released, and all OpenCV windows are closed.

#### The below code is written in Python and is designed to run in Google Colab. It includes a function for capturing an image from the webcam and then processes that image to detect faces and eyes, indicating whether the subject is awake or sleeping

# Sleeping Detector using Webcam in Google Colab

This document explains how to create a sleeping detector using OpenCV in Google Colab. The process involves capturing an image from the webcam and analyzing it to determine whether the subject is awake or sleeping based on face and eye detection.

## Code
# Importing necessary libraries
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# JavaScript to capture image from webcam
def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
          const div = document.createElement('div');
          const capture = document.createElement('button');
          capture.textContent = 'Capture';
          div.appendChild(capture);
          document.body.appendChild(div);

          const video = document.createElement('video');
          video.style.display = 'block';
          const stream = await navigator.mediaDevices.getUserMedia({video: true});
          document.body.appendChild(video);
          video.srcObject = stream;
          await video.play();

          // Resize video output
          google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

          // Wait for Capture button click
          await new Promise((resolve) => capture.onclick = resolve);

          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          stream.getTracks().forEach(track => track.stop());
          div.remove();
          video.remove();
          return canvas.toDataURL('image/jpeg', quality);
        }
        ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Capture the photo using webcam
filename = take_photo()
print(f"Saved to {filename}")

# Load the Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Read the captured image
img = cv2.imread(filename)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect face in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # Draw rectangle around the face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Region of interest (face area) for eye detection
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    if len(eyes) >= 2:
        cv2.putText(img, "Awake", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(img, "Sleeping", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw rectangles around eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# Display the result
cv2_imshow(img)
