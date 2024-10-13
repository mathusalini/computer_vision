![facefeatures](https://github.com/user-attachments/assets/7301a0f3-15c9-4573-b138-f7e3632488fa)

## Facial Feature Detection Using Haar Cascade

The diagram above visualizes how facial features such as eyes, nose, and mouth are detected using Haar-like features in a Haar Cascade algorithm. This process is essential for:

1. **Detecting the Position of a Face**: The Haar Cascade algorithm scans the image to locate the position of a face based on the identified features.
2. **Classifying Sleep Status**: Once the face is detected, the algorithm can analyze the state of the eyes to classify whether a person is asleep or awake.


- **Eyes**: The presence or absence of open eyes is a critical factor in determining if a person is sleeping.
- **Nose**: Helps in accurately positioning the face and recognizing facial orientation.
- **Mouth**: Also plays a role in understanding the overall facial expression.

```mermaid
flowchart TD
    A[Load HaarCascade Face Algorithm] 
    A1[Load the pre-trained HaarCascade classifier for face detection (e.g., haarcascade_frontalface_default.xml).]
    A --> A1

    B[Initialize Camera] 
    B1[Access the camera using OpenCV (e.g., cv2.VideoCapture(0) to initialize webcam).]
    B --> B1

    C[Read Frames from Camera] 
    C1[Continuously capture frames from the camera using a loop (e.g., ret, frame = cap.read()).]
    C --> C1

    D[Convert Color Image to Grayscale] 
    D1[Convert the captured frame to grayscale for processing (e.g., gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).]
    D --> D1

    E[Obtain Face Coordinates Using HaarCascade] 
    E1[Use the loaded HaarCascade model to detect faces in the grayscale image (e.g., faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)).]
    E --> E1

    F[Draw Rectangle on Face Coordinates] 
    F1[For each detected face, draw a rectangle around it on the original color frame (e.g., cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)).]
    F --> F1

    G[Display Output] 
    G1[Display the resulting frame with the drawn rectangle in a window (e.g., cv2.imshow('Face Detection', frame)).]
    G --> G1

