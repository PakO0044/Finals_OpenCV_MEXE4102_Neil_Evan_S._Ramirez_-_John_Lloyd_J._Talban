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
* code
   ```python
   !git clone https://github.com/PakO0044/Finals_OpenCV_MEXE4102_Neil_Evan_S._Ramirez_-_John_Lloyd_J._Talban.git
   %cd Finals_OpenCV_MEXE4102_Neil_Evan_S._Ramirez_-_John_Lloyd_J._Talban
   from IPython.display import clear_output
   clear_output()
   ```
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

   ```python
   import cv2
   import numpy as np
   from google.colab.patches import cv2_imshow

   img = np.zeros((512, 512, 3), np.uint8)
   #Write a Text
   cv2.putText(img,"NYAWWWWW",(35,254),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
   ```
   ![image](https://github.com/user-attachments/assets/0b9fb70c-0401-41f2-bb03-91809b764aeb)

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

  










