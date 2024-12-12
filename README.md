# **Introduction**
## Shapes That Fit: Geometric Foundations of Tessellation
* **Understanding Tessellation in Computer Vision:** Tessellation involves tiling a plane with one or more geometric shapes without overlaps or gaps. In computer vision, this problem extends to identifying, analyzing, and generating patterns in images or data.
* **Complexity:** Geometric tessellation demands efficient algorithms to detect, classify, and recreate intricate tiling patterns, especially when working with irregular or distorted shapes.
* **Applications:** Tessellation forms the basis for various tasks, including texture analysis, object recognition, and architectural design in images.
#### Significance in Computer Vision
* **Texture Analysis:** Tessellation aids in understanding surface properties, such as roughness or regularity, from images, which is crucial in fields like material science or quality control.
* **Pattern Recognition:** It supports identifying repeating geometric structures in natural or synthetic environments, which has applications in remote sensing and art restoration.
* **Image Compression:** Efficient tessellation algorithms help segment images into simpler geometric components, optimizing storage and processing requirements.
* **3D Reconstruction:** Tessellation techniques contribute to modeling and rendering complex surfaces in virtual environments or augmented reality.
* **Biological and Urban Mapping:** Analyzing natural patterns (e.g., honeycombs) or man-made layouts (e.g., city grids) often involves tessellation principles.

# Abstract
Summarize your project's objective, approach, and expected results.
# Project Methods
* Provide a step-by-step explanation of your methodology in bulleted form.
* Avoid paragraphs; focus on clarity and conciseness.
# Conclusion
* Summarize your findings, challenges, and outcomes.
# Additional Materials
* 16 Basic OpenCV projects

  ```python
   !git clone https://github.com/PakO0044/Finals_OpenCV_MEXE4102_Neil_Evan_S._Ramirez_-_John_Lloyd_J._Talban.git
   %cd Finals_OpenCV_MEXE4102_Neil_Evan_S._Ramirez_-_John_Lloyd_J._Talban
   from IPython.display import clear_output
   clear_output()
   ```
1. Converting Images to Grayscale
- Use the color space conversion code to convert RGB images to grayscale for basic image preprocessing.
  ```python
   import cv2
   from google.colab.patches import cv2_imshow

   #colorful image - 3 channels
   image = cv2.imread("Images/kobe.jpg")
   print(image.shape)

   #grayscale image
   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   print(gray.shape)
   cv2_imshow(gray)
   ```
   ![image](https://github.com/user-attachments/assets/c5e338e9-f31c-4cf5-b35f-d5010f3396a2)

2. Visualizing Edge Detection
- Apply the edge detection code to detect and visualize edges in a collection of object images.
  ```python
   import cv2
   from google.colab.patches import cv2_imshow
   import numpy as np

   image = cv2.imread("Images/motor.jpg")
   # cv2_imshow(image)

   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   canny_image = cv2.Canny(gray,150, 200)
   cv2_imshow(canny_image)
   ```
   ![image](https://github.com/user-attachments/assets/1e42cb03-53ce-4ba0-b9e1-598e53afb8f1)

3. Demonstrating Morphological Erosion
- Use the erosion code to show how an image's features shrink under different kernel sizes.
  ```python
   import cv2
   from google.colab.patches import cv2_imshow
   import numpy as np

   image = cv2.imread("Images/kobe.jpg")
   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   canny_image = cv2.Canny(gray,150, 200)
   kernel = np.ones((1,2), np.uint8)

   #Erosion
   erode_image = cv2.erode(canny_image,kernel, iterations=1)
   cv2_imshow(erode_image)
   ```
   ![image](https://github.com/user-attachments/assets/5cc1bc6a-a4b2-4a8f-9d40-592fcbdebc15)

4. Demonstrating Morphological Dilation
- Apply the dilation code to illustrate how small gaps in features are filled.
  ```python
   import cv2
   from google.colab.patches import cv2_imshow
   import numpy as np

   image = cv2.imread("Images/kobe.jpg")
   gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   canny_image = cv2.Canny(gray,150, 200)
   kernel = np.ones((5,5), np.uint8)
   #Dilation
   dilate_image = cv2.dilate(canny_image, kernel, iterations=1)
   cv2_imshow(dilate_image)
   ```
   ![image](https://github.com/user-attachments/assets/fc2d5d17-14c4-4530-9f44-700fe3079b1d)

5. Reducing Noise in Photos
- Use the denoising code to clean noisy images and compare the before-and-after effects.
  ```python
   import cv2
   from google.colab.patches import cv2_imshow
   import numpy as np

   image = cv2.imread("Images/kobe.jpg")
   # cv2_imshow(image)
   dst = cv2.fastNlMeansDenoisingColored(image, None, 50, 20, 7, 15)

   display = np.hstack((image, dst))
   cv2_imshow(display)
   ```
   ![image](https://github.com/user-attachments/assets/00dd7ffb-5de1-40f1-af64-94763bdbbeee)

6. Drawing Geometric Shapes on Images
- Apply the shape-drawing code to overlay circles, rectangles, and lines on sample photos.
  ```python
   import cv2
   import numpy as np
   from google.colab.patches import cv2_imshow

   img = np.zeros((512, 512, 3), np.uint8)
   #uint8: 0 to 255

   # Drawing Function
   # Draw a Circle
   cv2.circle(img, (100,100), 50, (0,255,0),5)
   # Draw a Rectangle
   cv2.rectangle(img,(200,200),(400,500),(0,0,255),5)
   # Displaying the Image
   cv2_imshow(img)
   ```
   ![image](https://github.com/user-attachments/assets/d9f2b9c5-ae95-4098-986a-ae7b8a35946a)

7. Adding Text to Images
- Use the text overlay code to label images with captions, annotations, or titles.
  ```python
   import cv2
   import numpy as np
   from google.colab.patches import cv2_imshow

   img = np.zeros((512, 512, 3), np.uint8)
   #Write a Text
   cv2.putText(img,"NYAWWWWW",(35,254),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
   ```
   ![image](https://github.com/user-attachments/assets/0b9fb70c-0401-41f2-bb03-91809b764aeb)

8. Isolating Objects by Color
- Apply the HSV thresholding code to extract and display objects of specific colors from an image.
  ```python
   import cv2
   import numpy as np
   from google.colab.patches import cv2_imshow
   #BGR Image . It is represented in Blue, Green and Red Channels...
   image = cv2.imread("Images/shapes.png")
   hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

   # Blue Color Triangle
   lower_hue = np.array([65,0,0])
   upper_hue = np.array([110, 255,255])

   # Red Color
  lower_hue = np.array([0,0,0])
  upper_hue = np.array([20,255, 255])

  # Green Color
  lower_hue = np.array([46,0,0])
  upper_hue = np.array([91,255,255])

  # Yellow Color
  lower_hue = np.array([21,0,0])
  upper_hue = np.array([45,255,255])

  mask = cv2.inRange(hsv,lower_hue,upper_hue)
  cv2_imshow(mask)
  result = cv2.bitwise_and(image, image, mask = mask)
  cv2_imshow(result)
  cv2_imshow(image)
  ```
  ![image](https://github.com/user-attachments/assets/2729a579-d760-4c0a-8924-e563baa43a52)

9. Detecting Faces in Group Photos
- Use the face detection code to identify and highlight faces in group pictures.
  ```python
  import cv2
  from google.colab.patches import cv2_imshow

  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  # img = cv2.imread("images/person.jpg")
  img = cv2.imread("Images/groupie.jpeg")
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
  faces = face_cascade.detectMultiScale(gray,1.3,5)
  # print(faces)
  for (x,y,w,h) in faces:
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

  cv2_imshow(img)
  ```
  ![image](https://github.com/user-attachments/assets/80409656-e6bf-412b-97f9-d314964aee94)

10. # Outlining Shapes with Contours
* Apply the contour detection code to outline and highlight shapes in simple object images.
  ```python
  import cv2
  import numpy as np
  from google.colab.patches import cv2_imshow

  img = cv2.imread("Images/shapes.png")
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  ret, thresh = cv2.threshold(gray,50,255,1)
  contours,h = cv2.findContours(thresh,1,2)
  # cv2_imshow(thresh)
  for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    n = len(approx)
    if n==6:
      # this is a hexagon
      print("We have a hexagon here")
      cv2.drawContours(img,[cnt],0,255,10)
    elif n==3:
      # this is a triangle
      print("We found a triangle")
      cv2.drawContours(img,[cnt],0,(0,255,0),3)
    elif n>9:
      # this is a circle
      print("We found a circle")
      cv2.drawContours(img,[cnt],0,(0,255,255),3)
    elif n==4:
      # this is a Square
      print("We found a square")
      cv2.drawContours(img,[cnt],0,(255,255,0),3)
  cv2_imshow(img)
  ```
  ![image](https://github.com/user-attachments/assets/ab8329f4-783d-40fb-8043-cd8e78e1ddf2)

11. # Tracking a Ball in a Video
* Use the HSV-based object detection code to track a colored ball in a recorded video.
   ```python
  import cv2
  import numpy as np
  from google.colab.patches import cv2_imshow
  import time  # For adding delays between frames

  # Initialize the video and variables
  ball = []
  cap = cv2.VideoCapture("Videos/video.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV and create a mask for the ball color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hue = np.array([21, 0, 0])  # Adjust for your ball's color
    upper_hue = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        try:
            # Calculate the center of the ball
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # Draw a circle at the center
            cv2.circle(frame, center, 10, (255, 0, 0), -1)
            ball.append(center)
        except ZeroDivisionError:
            pass

        # Draw the tracking path
        if len(ball) > 2:
            for i in range(1, len(ball)):
                cv2.line(frame, ball[i - 1], ball[i], (0, 0, 255), 5)

    # Display the frame in the notebook
    cv2_imshow(frame)

    # Add a small delay to simulate real-time playback
    time.sleep(0.05)

  cap.release()
  ```
  ![image](https://github.com/user-attachments/assets/8d6f0b6d-5390-4d43-9dbd-4fe17e6d4526)

12. # Highlighting Detected Faces
* Apply the Haar cascade face detection code to identify and highlight multiple faces in family or crowd photos.
``` python
!git clone https://github.com/PakO0044/Finals_OpenCV_MEXE4102_Neil_Evan_S._Ramirez_-_John_Lloyd_J._Talban.git
!pip install face_recognition
%cd Finals_OpenCV_MEXE4102_Neil_Evan_S._Ramirez_-_John_Lloyd_J._Talban
```
```python
import face_recognition
import numpy as np
from google.colab.patches import cv2_imshow
import cv2

# Creating the encoding profiles
face_1 = face_recognition.load_image_file("face/gelo.jpg")
face_1_encoding = face_recognition.face_encodings(face_1)[0]

face_2 = face_recognition.load_image_file("face/xavier.jpg")
face_2_encoding = face_recognition.face_encodings(face_2)[0]

face_3 = face_recognition.load_image_file("face/neil.jpg")
face_3_encoding = face_recognition.face_encodings(face_3)[0]

known_face_encodings = [
                        face_1_encoding,
                        face_2_encoding,
                        face_3_encoding
]

known_face_names = [
                    "Angelo",
                    "Xavier",
                    "Da Neil Ramirez"
]
```
  ```python

file_name = "face/neil 2.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)

  ```
  ![image](https://github.com/user-attachments/assets/1a8af874-394e-4039-98d2-8823f374e88e)

13. # Extracting Contours for Shape Analysis
* Use contour detection to analyze and outline geometric shapes in hand-drawn images.
  ```python
  import cv2
  from google.colab.patches import cv2_imshow
  import numpy as np

  # Read the input image
  image = cv2.imread("Images/handrawn.jpg")

  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply edge detection
  edges = cv2.Canny(gray, 50, 150)

  # Find contours
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Create a copy of the original image to draw contours
  contour_image = image.copy()

  # Draw the contours on the image
  cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

  # Analyze each contour and approximate the shape
  for contour in contours:
    # Approximate the contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Find the bounding rectangle to label the shape
    x, y, w, h = cv2.boundingRect(approx)

    # Determine the shape based on the number of vertices
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        # Check if the shape is square or rectangle
        aspect_ratio = float(w) / h
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif len(approx) > 4:
        shape = "Circle"
    else:
        shape = "Polygon"

    # Put the name of the shape on the image
    cv2.putText(contour_image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

  # Stack the original, edge-detected, and contour images for display
  stacked_result = np.hstack((cv2.resize(image, (300, 300)),
                            cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (300, 300)),
                            cv2.resize(contour_image, (300, 300))))

  # Display the results
  cv2_imshow(stacked_result)
  ```
  ![image](https://github.com/user-attachments/assets/e0bd35f4-17c1-464b-b6c7-4fd583ff7276)

14. # Applying Image Blurring Techniques
* Demonstrate various image blurring methods (Gaussian blur, median blur) to soften details in an image.
  ```python
  import cv2
  from google.colab.patches import cv2_imshow
  import numpy as np

  image = cv2.imread("Images/motor.jpg")
  Gaussian = cv2.GaussianBlur(image,(7,7),0)
  Median = cv2.medianBlur(image,5)

  display = np.hstack((Gaussian,Median))
  cv2_imshow(display)
  ```
  ![image](https://github.com/user-attachments/assets/781476bc-4a57-480f-bdd3-a4d5fc7df885)

15. # Segmenting Images Based on Contours
* Use contour detection to separate different sections of an image, like dividing a painting into its distinct elements.
  ```python
  import cv2
  from google.colab.patches import cv2_imshow
  import numpy as np

  # Read the input image
  image = cv2.imread("Images/handrawn.jpg")

  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply binary thresholding
  _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

  # Find contours
  contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Create a blank mask for segmentation
  segmented_image = np.zeros_like(image)

  # Loop through each contour to extract and display segmented areas
  for i, contour in enumerate(contours):
    # Create a mask for the current contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Extract the segment by masking the original image
    segmented_part = cv2.bitwise_and(image, image, mask=mask)

    # Add the segment to the segmented image
    segmented_image = cv2.add(segmented_image, segmented_part)

    # Optionally draw bounding boxes for visualization
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


  # Display results
  cv2_imshow(image)  # Original image with bounding boxes
  cv2_imshow(segmented_image)  # Segmented image
  ```
  ![image](https://github.com/user-attachments/assets/52cd3810-6006-4f3b-b6ff-3b00a6779b9e)

16. # Combining Erosion and Dilation for Feature Refinement
* Apply erosion followed by dilation on an image to refine and smooth out small features.
  ```python
  import cv2
  from google.colab.patches import cv2_imshow
  import numpy as np

  image = cv2.imread("Images/motor.jpg")
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  canny_image = cv2.Canny(gray,150, 200)
  kernel = np.ones((1,1), np.uint8)
  erode_image = cv2.erode(canny_image,kernel, iterations=1)
  kernel1 = np.ones((3,3), np.uint8)
  dilate_image = cv2.dilate(erode_image, kernel1, iterations=1)

  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(canny_image, 'Canny Image', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(erode_image, 'Eroded', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(dilate_image, 'Feature Refined', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

  display = np.hstack((canny_image,erode_image,dilate_image))
  cv2_imshow(display)
  ```
  ![image](https://github.com/user-attachments/assets/1cb3371f-39d3-4fd6-bc0d-e20c00700779)

  
















