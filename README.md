<div align="center">

# **Shapes That Fit: Geometric Foundations of Tessellation**
## **Final Project: Open CV**
### MEXE 402 - MEXE 4102

</div>


# Table of Contents

1. [Introduction](#introduction)
2. [Abstract](#abstract)
3. [Project Methods](#project-methods)
4. [Conclusion](#conclusion)
5. [Part 1: 16 Basic OpenCV projects](#part-1-16-basic-opencv-projects)
6. [Part 2: Shapes That Fit: Geometric Foundations of Tessellation](#part-2-shapes-that-fit-geometric-foundations-of-tessellation)
7. [Part 2: Shapes That Fit: Geometric Foundations of Tessellation without a Dataset](#shapes-that-fit-geometric-foundations-of-tessellation-without-a-dataset)
<div align="justify">
8. [References](#references)
# **Introduction**

## Tessellation
* Tessellation involves tiling a plane with geometric shapes without overlaps or gaps.

#### **Significance in Computer Vision**
* **Texture Analysis:** It helps in understanding surface properties like roughness or regularity from images, crucial in fields such as material science.
* **Pattern Recognition:** Tessellation aids in identifying repeating geometric structures in images, applicable in fields like remote sensing and art restoration.
* **Image Compression:** Efficient tessellation algorithms assist in segmenting images into simpler geometric components, optimizing storage and processing requirements.
* **3D Reconstruction:** Tessellation techniques contribute to modeling and rendering complex surfaces in virtual environments or augmented reality.
* **Biological and Urban Mapping:** It involves analyzing natural patterns (e.g., honeycombs) or man-made layouts (e.g., city grids) using tessellation principles.

# Abstract

The project's primary focus is on creating intricate tessellation patterns using geometric shapes, aiming to address challenges in achieving desired aesthetic allure within the designs through a systematic approach. By meticulously integrating geometric shapes, the project seeks to generate visually captivating and harmonious tessellation designs as its core objective. Additionally, the project incorporates advanced features that significantly enhance its capabilities, including precise drawing mechanisms for overlaying shapes onto the canvas and the ability to adjust transparency levels, adding depth and sophistication to the overall visual impact of the tessellations. This abstract offers a comprehensive overview of the project's methods and expected outcomes, shedding light on the detailed process involved in crafting mesmerizing tessellation patterns that are both visually appealing and artistically compelling, showcasing the project's dedication to creating intricate and visually striking geometric compositions.

# Project Methods

#### **Import Libraries**
- **Libraries**:
  - `cv2`: For image processing.
  - `numpy`: For array manipulation.
  - `google.colab.patches.cv2_imshow`: To display images in Google Colab.

---

#### **Load Images**
- Use `cv2.imread()` to load the images for the triangle, circle, and square.
- Ensure the images are loaded with transparency support (`IMREAD_UNCHANGED`).

---

#### **Remove Background**
- Define a `remove_background()` function to make the background of each image transparent:
  - **Input Check**: Raise an error if the image fails to load.
  - **Alpha Channel Check**: If the image already has an alpha channel, return it as is.
  - **Add Alpha Channel**:
    - Split the image into blue, green, and red channels.
    - Create a new alpha channel initialized to fully opaque (`255`).
    - Define a white color range to identify the background.
    - Set the alpha channel to `0` (transparent) for pixels within the background color range.
  - Merge the color channels with the new alpha channel and return the image.

---

#### **Resize Images**
- Resize all shapes (triangle, circle, square) to a uniform size (`30x30 pixels`) using `cv2.resize()` to maintain consistency in tessellation.

---

#### **Create a Canvas**
- Initialize a blank canvas (`800x600 pixels`) with a white background using `np.ones()`.

---

#### **Define Image Overlay Function**
- Define `overlay_image()` to blend shapes onto the canvas:
  - Check if the overlay image has an alpha channel.
  - Blend the shape onto the canvas using the alpha channel for transparency.
  - If no alpha channel exists, directly overlay the image onto the canvas.

---

#### **Define Tessellation Parameters**
- Calculate the number of rows and columns:
  - `rows = canvas height // shape size`.
  - `cols = canvas width // shape size`.

---

#### **Create Tessellation Pattern**
- Use nested loops to iterate through rows and columns:
  - Compute the `(i + j) % 3` value to alternate shapes:
    - `0`: Place a triangle.
    - `1`: Place a circle.
    - `2`: Place a square.
  - Calculate the top-left position `(x, y)` for each shape and use `overlay_image()` to draw it on the canvas.

---

#### **Display and Save the Output**
- Use `cv2_imshow()` to display the tessellation in Colab.
- Optionally save the output image to a file (`tessellation_output.png`) using `cv2.imwrite()`.

---

# Conclusion

#### **Findings:** 
* Creation of tessellation patterns with triangles, circles, and squares.
* The code successfully integrates different geometric shapes to form a tessellation pattern on the canvas.

#### **Challenges:**
* Difficulty in achieving desired aesthetic appearance.
* Balancing symmetry and color coordination within the tessellation posed a significant challenge.

#### **Outcomes:** 
* Successful generation of tessellation patterns with alternating shapes and colors.
* The code effectively produces visually appealing tessellation patterns with a harmonious blend of shapes and colors.

#### **Additional Features:**
* Functions for drawing, overlaying shapes, and adjusting transparency.
* Essential functions are included for drawing shapes, overlaying them on the canvas with transparency adjustments, and ensuring seamless integration within the tessellation.

</div>


# Additional Materials
# Part 1: 16 Basic OpenCV projects

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

# Load the image
image = cv2.imread("Images/kobe.jpg")

# Write a Text
cv2.putText(image, "NYAWWWWW", (35, 254), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

# Display the image with text
cv2_imshow(image)
```
![image](https://github.com/user-attachments/assets/38dffc2e-c44c-4ff0-a623-f8a659f3a553)


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
![image](https://github.com/user-attachments/assets/83b2ec25-9676-4e89-8842-c1db6874b3f7)


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
![image](https://github.com/user-attachments/assets/bd377eaa-43e1-44fb-9f2a-3b3af4e1ba47)


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

  
# Part 2: Shapes That Fit: Geometric Foundations of Tessellation
``` python
!git clone https://github.com/PakO0044/Finals_OpenCV_MEXE4102_Neil_Evan_S._Ramirez_-_John_Lloyd_J._Talban.git
%cd Finals_OpenCV_MEXE4102_Neil_Evan_S._Ramirez_-_John_Lloyd_J._Talban
from IPython.display import clear_output
clear_output()

```
``` python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images in Colab

# Load the uploaded image for square (update the path to your dataset in Colab environment)
square = cv2.imread("Dataset/square.png", cv2.IMREAD_UNCHANGED)

# Function to remove the background by making it transparent
def remove_background(image):
    if image is None:
        raise ValueError("Image not found. Please check the file path.")
    if image.shape[2] == 4:  # If the image already has an alpha channel
        return image
    else:
        # Create an alpha channel (transparency) for the image
        b, g, r = cv2.split(image)
        alpha_channel = np.ones_like(b) * 255
        lower = np.array([240, 240, 240])  # White background range
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(image, lower, upper)
        alpha_channel[mask == 255] = 0  # Make the background transparent
        return cv2.merge((b, g, r, alpha_channel))

# Remove the background from the square image
try:
    square = remove_background(square)
except ValueError as e:
    print(e)

# Resize the square image for uniform tessellation
shape_size = 30  # Fixed size for each shape
square = cv2.resize(square, (shape_size, shape_size))

# Canvas dimensions
width, height = 800, 600
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas

# Function to overlay an image with transparency
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if overlay.shape[2] == 4:  # If the overlay has an alpha channel
        alpha_channel = overlay[:, :, 3] / 255.0
        for c in range(3):  # Blend each color channel
            background[y:y+h, x:x+w, c] = (
                alpha_channel * overlay[:, :, c] +
                (1 - alpha_channel) * background[y:y+h, x:x+w, c]
            )

# Tessellation parameters
rows, cols = height // shape_size, width // shape_size

# Create tessellation by placing only squares
for i in range(rows):
    for j in range(cols):
        x, y = j * shape_size, i * shape_size
        overlay_image(canvas, square, x, y)

# Display the tessellation
cv2_imshow(canvas)  # Use cv2_imshow for displaying in Colab
cv2.imwrite("square_tessellation.png", canvas)  # Optionally save the output for download
```
![image](https://github.com/user-attachments/assets/8ab930a7-5f7a-4e49-aa46-16d2fd7d6336)

``` python
# Import necessary libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images in Colab

# Load the triangle image (update the path for Colab)
triangle = cv2.imread("Dataset/triangle.png", cv2.IMREAD_UNCHANGED)

# Function to remove the background by making it transparent
def remove_background(image):
    if image is None:
        raise ValueError("Image not found. Please check the file path.")
    if image.shape[2] == 4:  # If the image already has an alpha channel
        return image
    else:
        # Create an alpha channel (transparency) for the image
        b, g, r = cv2.split(image)
        alpha_channel = np.ones_like(b) * 255
        lower = np.array([240, 240, 240])  # White background range
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(image, lower, upper)
        alpha_channel[mask == 255] = 0  # Make the background transparent
        return cv2.merge((b, g, r, alpha_channel))

# Remove background from the triangle image
try:
    triangle = remove_background(triangle)
except ValueError as e:
    print(e)

# Resize the triangle for uniform tessellation
shape_size = 50  # Triangle's side length
triangle = cv2.resize(triangle, (shape_size, shape_size))

# Create a flipped version of the triangle
triangle_flipped = cv2.flip(triangle, 0)

# Canvas dimensions
width, height = 800, 600
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas

# Function to overlay an image with transparency
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    # Ensure the overlay fits within the canvas
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return  # Skip if the overlay exceeds the canvas boundaries
    alpha_channel = overlay[:, :, 3] / 255.0  # Alpha transparency
    for c in range(3):  # Blend each color channel
        background[y:y+h, x:x+w, c] = (
            alpha_channel * overlay[:, :, c] +
            (1 - alpha_channel) * background[y:y+h, x:x+w, c]
        )

# Tessellation parameters
row_height = shape_size // 2  # Row spacing (half triangle height)
cols = width // shape_size

# Create tessellation with alternating triangles
for i in range(height // row_height):
    for j in range(cols):
        x = j * shape_size
        y = i * row_height
        # Offset every second row to create the tessellation
        if i % 2 == 0:
            overlay_image(canvas, triangle if j % 2 == 0 else triangle_flipped, x, y)
        else:
            offset_x = shape_size // 2  # Shift odd rows
            overlay_image(canvas, triangle if j % 2 == 0 else triangle_flipped, x + offset_x, y)

# Display the tessellation
cv2_imshow(canvas)  # Use cv2_imshow for displaying in Colab
cv2.imwrite("triangle_tessellation.png", canvas)  # Optionally save the output for download
```
![image](https://github.com/user-attachments/assets/9a647d33-b4c7-4c20-8b64-24f1708530ac)

``` python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images in Colab

# Load the images
triangle = cv2.imread("Dataset/triangle.png", cv2.IMREAD_UNCHANGED)
square = cv2.imread("Dataset/square.png", cv2.IMREAD_UNCHANGED)

# Function to remove the background by making it transparent
def remove_background(image):
    if image is None:
        raise ValueError("Image not found. Please check the file path.")
    if image.shape[2] == 4:
        return image
    else:
        b, g, r = cv2.split(image)
        alpha_channel = np.ones_like(b) * 255
        lower = np.array([240, 240, 240])  # White background range
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(image, lower, upper)
        alpha_channel[mask == 255] = 0
        return cv2.merge((b, g, r, alpha_channel))

# Remove backgrounds from the images
try:
    triangle = remove_background(triangle)
    square = remove_background(square)
except ValueError as e:
    print(e)

# Resize the shapes for uniform tessellation
shape_size = 50  # Size for the shapes
triangle = cv2.resize(triangle, (shape_size, shape_size))
square = cv2.resize(square, (shape_size, shape_size))

# Create a flipped version of the triangle
triangle_flipped = cv2.flip(triangle, 0)

# Canvas dimensions
width, height = 800, 600
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas

# Function to overlay an image with transparency
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return  # Skip if the overlay exceeds the canvas boundaries
    alpha_channel = overlay[:, :, 3] / 255.0  # Alpha transparency
    for c in range(3):  # Blend each color channel
        background[y:y+h, x:x+w, c] = (
            alpha_channel * overlay[:, :, c] +
            (1 - alpha_channel) * background[y:y+h, x:x+w, c]
        )

# Tessellation parameters
row_height = shape_size // 2  # Row spacing (half triangle height)
cols = width // shape_size

# Create tessellation with alternating shapes
for i in range(height // row_height):
    for j in range(cols):
        x = j * shape_size
        y = i * row_height
        if i % 2 == 0:  # Even rows
            if j % 2 == 0:
                overlay_image(canvas, triangle, x, y)
            else:
                overlay_image(canvas, square, x, y)
        else:  # Odd rows
            offset_x = shape_size // 2  # Shift odd rows
            if j % 2 == 0:
                overlay_image(canvas, square, x + offset_x, y)
            else:
                overlay_image(canvas, triangle_flipped, x + offset_x, y)

# Display the tessellation
cv2_imshow(canvas)  # Use cv2_imshow for displaying in Colab
cv2.imwrite("triangle_square_tessellation.png", canvas)  # Optionally save the output
```
![image](https://github.com/user-attachments/assets/7fb34eed-ad0f-4220-ace3-553f04645f53)

``` python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images in Colab

# Load the uploaded images
triangle = cv2.imread("Dataset/triangle.png", cv2.IMREAD_UNCHANGED)
circle = cv2.imread("Dataset/circle.png", cv2.IMREAD_UNCHANGED)
square = cv2.imread("Dataset/square.png", cv2.IMREAD_UNCHANGED)

# Function to remove the background by making it transparent
def remove_background(image):
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")
    if image.shape[2] == 4:  # Already has an alpha channel
        return image
    else:
        # Create an alpha channel for the image
        b, g, r = cv2.split(image)
        alpha_channel = np.ones_like(b) * 255
        # Define background color range (white background assumed)
        lower = np.array([240, 240, 240])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(image, lower, upper)
        alpha_channel[mask == 255] = 0  # Set transparent for the background
        return cv2.merge((b, g, r, alpha_channel))

# Remove the background from the images
try:
    triangle = remove_background(triangle)
    circle = remove_background(circle)
    square = remove_background(square)
except ValueError as e:
    print(e)

# Resize the images for uniformity in tessellation
shape_size = 30  # Fixed size for each shape
triangle = cv2.resize(triangle, (shape_size, shape_size))
circle = cv2.resize(circle, (shape_size, shape_size))
square = cv2.resize(square, (shape_size, shape_size))

# Canvas dimensions
width, height = 800, 600
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas

# Function to overlay an image with transparency
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if overlay.shape[2] == 4:  # If the overlay has an alpha channel
        alpha_channel = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha_channel * overlay[:, :, c] +
                (1 - alpha_channel) * background[y:y+h, x:x+w, c]
            )
    else:
        background[y:y+h, x:x+w] = overlay

# Tessellation parameters
rows, cols = height // shape_size, width // shape_size

# Create tessellation by alternating shapes
for i in range(rows):
    for j in range(cols):
        x, y = j * shape_size, i * shape_size
        if (i + j) % 3 == 0:
            overlay_image(canvas, triangle, x, y)
        elif (i + j) % 3 == 1:
            overlay_image(canvas, circle, x, y)
        else:
            overlay_image(canvas, square, x, y)

# Display the tessellation
cv2_imshow(canvas)  # Use cv2_imshow to display the image in Colab
cv2.imwrite("tessellation_output.png", canvas)  # Optionally save the output
```
![image](https://github.com/user-attachments/assets/ded40196-76b8-46f2-addc-1d7ce6fb0be0)

# Part 2: Shapes That Fit: Geometric Foundations of Tessellation without a Dataset
``` python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Create a blank canvas
img = np.zeros((512, 512, 3), np.uint8)

# Tessellation parameters
canvas_height, canvas_width = img.shape[:2]
shape_size = 50  # Size of each shape (side length for square)
spacing = 60     # Spacing between shapes

# Function to draw a square
def draw_square(image, center, size, color):
    half_size = size // 2
    top_left = (center[0] - half_size, center[1] - half_size)
    bottom_right = (center[0] + half_size, center[1] + half_size)
    cv2.rectangle(image, top_left, bottom_right, color, -1)  # Fill the square with the specified color

# Draw tessellation pattern with alternating color of squares
for row in range(0, canvas_height, spacing):
    for col in range(0, canvas_width, spacing):
        center = (col + shape_size // 2, row + shape_size // 2)
        if (row + col) % (2 * spacing) == 0:
            # Draw a red square
            draw_square(img, center, shape_size, (255, 0, 0))
        else:
            # Draw a green square
            draw_square(img, center, shape_size, (0, 255, 0))

# Display the tessellated pattern with alternating color of squares
cv2_imshow(img)
```
![image](https://github.com/user-attachments/assets/9a699109-f9cc-413d-be74-9b924b94fe6c)

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Read the uploaded image
img = np.zeros((512, 512, 3), np.uint8)

# Tessellation parameters
canvas_height, canvas_width = img.shape[:2]
shape_size = 50  # Size of each shape (side length for triangle)
spacing = 60     # Spacing between shapes

# Function to draw a triangle
def draw_triangle(image, center, size, color, thickness, direction="up"):
    half_size = size // 2
    if direction == "up":
        pt1 = (center[0], center[1] - half_size)  # Top vertex
        pt2 = (center[0] - half_size, center[1] + half_size)  # Bottom-left vertex
        pt3 = (center[0] + half_size, center[1] + half_size)  # Bottom-right vertex
    else:  # Downward-facing triangle
        pt1 = (center[0], center[1] + half_size)  # Bottom vertex
        pt2 = (center[0] - half_size, center[1] - half_size)  # Top-left vertex
        pt3 = (center[0] + half_size, center[1] - half_size)  # Top-right vertex
    points = np.array([pt1, pt2, pt3], np.int32)
    cv2.fillPoly(image, [points], color)

# Draw tessellation pattern with alternating color of triangles
for row in range(0, canvas_height, spacing):
    for col in range(0, canvas_width, spacing):
        center = (col + shape_size // 2, row + shape_size // 2)
        if (row + col) % (2 * spacing) == 0:
            # Draw a green triangle
            draw_triangle(img, center, shape_size, (0, 255, 0), -1)
        else:
            # Draw a red triangle
            draw_triangle(img, center, shape_size, (255, 0, 0), -1, direction="down")

# Display the tessellated pattern with alternating color of triangles
cv2_imshow(img)
```
![image](https://github.com/user-attachments/assets/5c2a87a0-7e52-4cd5-b78c-c3eb88a6a214)

``` python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Create a blank canvas
img = np.zeros((512, 512, 3), np.uint8)

# Tessellation parameters
canvas_height, canvas_width = img.shape[:2]
shape_size = 50  # Size of each shape (side length for square)
spacing = 60     # Spacing between shapes

# Function to draw a square
def draw_square(image, center, size, color):
    half_size = size // 2
    top_left = (center[0] - half_size, center[1] - half_size)
    bottom_right = (center[0] + half_size, center[1] + half_size)
    cv2.rectangle(image, top_left, bottom_right, color, -1)  # Fill the square with the specified color

# Function to draw a triangle
def draw_triangle(image, center, size, color, direction="up"):
    half_size = size // 2
    if direction == "up":
        pt1 = (center[0], center[1] - half_size)  # Top vertex
        pt2 = (center[0] - half_size, center[1] + half_size)  # Bottom-left vertex
        pt3 = (center[0] + half_size, center[1] + half_size)  # Bottom-right vertex
    else:  # Downward-facing triangle
        pt1 = (center[0], center[1] + half_size)  # Bottom vertex
        pt2 = (center[0] - half_size, center[1] - half_size)  # Top-left vertex
        pt3 = (center[0] + half_size, center[1] - half_size)  # Top-right vertex
    points = np.array([pt1, pt2, pt3], np.int32)
    cv2.fillPoly(image, [points], color)

# Draw tessellation pattern with alternating color of squares and triangles
for row in range(0, canvas_height, spacing):
    for col in range(0, canvas_width, spacing):
        center = (col + shape_size // 2, row + shape_size // 2)
        if (row + col) % (2 * spacing) == 0:
            # Draw a red square
            draw_square(img, center, shape_size, (255, 0, 0))
        else:
            # Draw a green triangle
            draw_triangle(img, center, shape_size, (0, 255, 0))

# Display the tessellated pattern with alternating color of squares and triangles
cv2_imshow(img)
```
![image](https://github.com/user-attachments/assets/091de822-052d-461f-a1ce-1042c108bc32)

``` python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Create a blank canvas
img = np.zeros((512, 512, 3), np.uint8)

# Tessellation parameters
canvas_height, canvas_width = img.shape[:2]
shape_size = 50  # Size of each shape (side length for circle)
spacing = 60     # Spacing between shapes

# Function to draw a circle
def draw_circle(image, center, radius, color):
    cv2.circle(image, center, radius, color, -1)

# Function to draw a square
def draw_square(image, center, size, color):
    half_size = size // 2
    top_left = (center[0] - half_size, center[1] - half_size)
    bottom_right = (center[0] + half_size, center[1] + half_size)
    cv2.rectangle(image, top_left, bottom_right, color, -1)

# Function to draw a triangle
def draw_triangle(image, center, size, color, direction="down"):
    half_size = size // 2
    if direction == "up":
        pt1 = (center[0], center[1] - half_size)  # Top vertex
        pt2 = (center[0] - half_size, center[1] + half_size)  # Bottom-left vertex
        pt3 = (center[0] + half_size, center[1] + half_size)  # Bottom-right vertex
    else:  # Downward-facing triangle
        pt1 = (center[0], center[1] + half_size)  # Bottom vertex
        pt2 = (center[0] - half_size, center[1] - half_size)  # Top-left vertex
        pt3 = (center[0] + half_size, center[1] - half_size)  # Top-right vertex
    points = np.array([pt1, pt2, pt3], np.int32)
    cv2.fillPoly(image, [points], color)

# Draw tessellation pattern with alternating color of shapes
for row in range(0, canvas_height, spacing):
    for col in range(0, canvas_width, spacing):
        center = (col + shape_size // 2, row + shape_size // 2)
        if (row + col) % (2 * spacing) == 0:
            # Draw a red square
            draw_square(img, center, shape_size, (255, 0, 0))
        elif (row + col) % (3 * spacing) == 0:
            # Draw a green triangle
            draw_triangle(img, center, shape_size, (0, 255, 0))
        else:
            # Draw a blue circle
            draw_circle(img, center, shape_size // 2, (0, 0, 255))

# Display the tessellated pattern with alternating shapes
cv2_imshow(img)
```
![image](https://github.com/user-attachments/assets/cf1190ad-6dc1-4913-849f-eca51d7e56a6)

# References
- https://www.kaggle.com/datasets/singhnavjot2062001/geometric-shapes-circle-square-triangle
- https://mathworld.wolfram.com/RegularTessellation.html
- https://mathworld.wolfram.com/EquilateralTriangle.html
- https://mathworld.wolfram.com/Square.html
- https://youtu.be/E3Lg4aZVCAU?si=_UT8JlruRT_q7wi1










