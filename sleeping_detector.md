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


If we did this in Google Colab, the code will come like this.
![screenshot_code](https://github.com/user-attachments/assets/1fa5d988-43bb-45bc-93e8-b1059c3e0f07)

after this webcam capture a image,then run the below code,

![Screenshot 2024-10-13 124938](https://github.com/user-attachments/assets/fd5e6f24-c4b5-4496-90e4-acbabbe853e8)
We can detect whether it is sleeping or awake😊