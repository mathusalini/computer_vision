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

# Sleeping Detector Code
![sleeping_detector_code](https://github.com/user-attachments/assets/186d35cc-e847-45af-87d6-4c40c12aac4d)

```
import cv2
import numpy as np
# Load the pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Function to detect eyes and classify sleep/awake based on eye detection
def detect_sleep(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect face in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest (face area) for eye detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            # Eyes detected -> Awake
            cv2.putText(frame, "Awake", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Less than two eyes detected -> Sleeping
            cv2.putText(frame, "Sleeping", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return frame

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame not captured correctly, break the loop
    if not ret:
        break
    
    # Call the sleep detection function
    frame = detect_sleep(frame)
    
    # Display the resulting frame
    cv2.imshow('Sleeping Detector', frame)
    
    # Press 'q' to exit the webcam window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
```



## Using google colab
If we did this in Google Colab, the code will come like this.
![screenshot_code](https://github.com/user-attachments/assets/1fa5d988-43bb-45bc-93e8-b1059c3e0f07)

webcam capture a image,then run the below code,

![Screenshot 2024-10-13 124938](https://github.com/user-attachments/assets/fd5e6f24-c4b5-4496-90e4-acbabbe853e8)


We can detect whether if anyone sleep😊

## code
```
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
```
```
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
```