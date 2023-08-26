#importing necessary packages
import cv2 as cv
import numpy as np

#reading an image
def read_image(image_path:str, window_name:str):
    print('Task has started')
    print('Press any key to cancel')
    #Load an image from the given path
    img = cv.imread(image_path)
    #Displying the loaded image
    cv.imshow(window_name, img)
    #Additional functions for ensuring productivity
    cv.waitKey(0)
    print('Task has ended')
    cv.destroyAllWindows()

def read_video(video_path:str, window_name:str):
    #Load a video from the given path
    capture = cv.VideoCapture(video_path)
    print('Task has started')
    print("Press 'x' key to cancel")
    #Calling a loop
    while True:
        #loading a frame from the video
        ret, frame = capture.read()
        # If there are no more frames to read, break out of the loop
        if not ret:
            print('All frames in the Video are succesfully played --- No frames are there/left to read in the path specified')
            break
        #Displying the loaded frame
        cv.imshow(window_name, frame)
        #Giving an opportunity for the user to cancel the process
        if cv.waitKey(20)&0xFF==ord('x'):
            print ('Operation cancelled by the user')
            break
    print('Task has ended')
    #Relasing the Video file
    capture.release()
    #Additional functions for ensuring productivity
    cv.destroyAllWindows()

   #rescaling the image
def rescale_image(image_path:str, scale:float, window_name:str):
    print('Task has started')
    print('Press any key to cancel')
    #reading an image
    frame = cv.imread(image_path)
    #Setting the width
    width = int(frame.shape[1]*scale)
    #Setting the height
    height = int(frame.shape[0]*scale)
    #Combining them into a tuple
    dimensions = (width, height)
    #Using a predefined function from opencv to rescale the image into a variable
    resized_image = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    #Displying the rescaled image
    cv.imshow(window_name, resized_image)
    #Additional functions for ensuring productivity
    cv.waitKey(0)
    print('Task has ended')
    cv.destroyAllWindows()

#rescaling the video
def rescale_video(video_path:str, window_name:str, scale:float):
    #Reading the video
    cap = cv.VideoCapture(video_path)
    print('Task has started')
    print("Press 'x' key to cancel")
    #Calling a loop
    while True:
        #loading a frame from the video
        ret, frame = cap.read()
        if not ret:
            # If there are no more frames to read, break out of the loop
            print('All frames in the Video are succesfully played --- No frames are there/left to read in the path specified')
            break
        #Setting the width
        width = int(frame.shape[1] * scale)
        #Setting the height
        height = int(frame.shape[0] * scale)
        #Combining them into a tuple
        dimensions = (width, height)
        #Using a predefined function from opencv to rescale the video into a variable
        resized_frame = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
        #Displying the rescaled video
        cv.imshow(window_name, resized_frame)
        #Giving an opportunity for the user to cancel the process
        if cv.waitKey(20) & 0xFF == ord('x'):
            print('Operation cancelled by the user')
            break
    print('Task has ended')
    #Relasing the Video file
    cap.release()
    #Additional functions for ensuring productivity
    cv.destroyAllWindows()

#painting an image
def paint_image(color, window_name:str):
    blank = np.zeros((500,500,3), dtype='uint8')
    #setting a color to the specific region
    blank[200:300,300:400] = color
    #Displaing the Output image
    cv.imshow (window_name, blank)
    #Additional functions for ensuring productivity
    cv.waitKey(0)
    cv.destroyAllWindows()

#Draws a rectangle
def draw_rectangle(color, outline):
    blank = np.zeros((500,500,3), dtype='uint8')
    #Drawing a rectangle using OpenCV's function
    cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), color, thickness=outline)
    #Displaying the output image
    cv.imshow('Rectangle', blank)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Draws a circle
def draw_circle(color, outline, radius):
    blank = np.zeros((500,500,3), dtype='uint8')
    #Drawing a circle using OpenCV's function
    cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), radius, color, thickness=outline)
    #Displaying the output image
    cv.imshow('Circle', blank)
    cv.waitKey(0)

#Draws a line
def draw_line(color, outline):
    blank = np.zeros((500,500,3), dtype='uint8')
    cv.line(blank, (0,0), (300,400), color, outline)
    #Displaying the output image
    cv.imshow('Line', blank)
    cv.waitKey(0)

#Writes text
def write_text(text, color, outline):
    blank = np.zeros((500,500,3), dtype='uint8')
    cv.putText(blank, text, (0,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, outline)
    #Displaying the output image
    cv.imshow('Text', blank)
    cv.waitKey(0)

#Shifts the position of the image
def image_translate(image_path):
    img = cv.imread(image_path, 0)
    #creating a list
    rows, cols = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv.warpAffine(img, M, (cols,rows))
    #Displaying the output image
    cv.imshow('Translated Image', dst) 
    cv.waitKey(0) 

#Flipping an image Vertically
def img_flipV(image_path):
    img = cv.imread(image_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 0], [0, -1, rows],[0, 0, 1]])
    reflected_img = cv.warpPerspective(img, M,(int(cols),int(rows)))
    #Displaying the output image
    cv.imshow('Vertically flipped Image', reflected_img)
    cv.imwrite('Saved Files/Vertically flipped Image.jpg', reflected_img)
    cv.waitKey(0)

#Flipping an image Horizontally
def img_flipH(image_path):
    img = cv.imread(image_path, 0)
    rows, cols = img.shape
    M = np.float32([[-1, 0, cols], [0, 1, 0], [0, 0, 1]])
    reflected_img = cv.warpPerspective(img, M,(int(cols),int(rows)))
    #Displaying the output image
    cv.imshow('Horizontally flipped Image', reflected_img)
    cv.imwrite('Saved Files/Horizontally flipped Image.jpg', reflected_img)
    cv.waitKey(0)

#Rotating an Image
def r_img(image_path):
    img = cv.imread(image_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
    img_rotation = cv.warpAffine(img, cv.getRotationMatrix2D((cols/2, rows/2), 30, 0.6), (cols, rows))
    #Displaying the output image
    cv.imshow('Rotated Image', img_rotation)
    cv.imwrite('Saved Files/Rotated Image.jpg', img_rotation)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Shrinking an Image
def s_img(image_path):
    img = cv.imread(image_path)
    img_shrinked = cv.resize(img, (350, 300), interpolation = cv.INTER_AREA)
    #Displaying the output image
    cv.imshow('Shrinked Image', img_shrinked)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Enlarging an Image
def e_img(image_path):
    img = cv.imread(image_path)
    img_enlarged = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
    #Displaying the output image
    cv.imshow('Enlarged Image', img_enlarged)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Cropping an image
def c_img(image_path):
    img = cv.imread(image_path, 0)
    cropped_img = img[100:300, 100:300]
    cv.imwrite('Saved Files/Cropped Image.jpg', cropped_img)
    #Displaying the output image
    cv.imshow('Cropped Image', cropped_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Shearing an Image in X-axis
def SHEAR_img_X(image_path):
    img = cv.imread(image_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
    #Displaying the output image
    cv.imshow('Sheared X-axis', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Shearing an Image in Y-axis
def SHEAR_img_Y(image_path):
    img = cv.imread(image_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]])
    sheared_img = cv.warpPerspective(img, M,
    (int(cols*1.5), int(rows*1.5)))
    #Displaying the output image
    cv.imshow('Sheared Y-axis', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Blurring an image using 2D Convolution method
def BLUR_2DKconv(image_path):
    # Reading the image
    image = cv.imread(image_path)
    # Creating the kernel with numpy
    kernel2 = np.ones((5, 5), np.float32)/25
    # Applying the filter
    img = cv.filter2D(src=image, ddepth=-1, kernel=kernel2)
    # showing the image
    cv.imshow('Original', image)
    cv.imshow('Kernel Blur', img)
    cv.waitKey()
    cv.destroyAllWindows()

#Blurring an image using Average Blur
def BLUR_Avg(image_path):
    image = cv.imread(image_path)
    # Applying the filter
    averageBlur = cv.blur(image, (5, 5))
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Average blur', averageBlur)
    cv.waitKey()
    cv.destroyAllWindows() 

#Blurruing an image using Gaussian Blur
def BLUR_Gaus(image_path):
    image = cv.imread(image_path)
    # Applying the filter
    gaussian = cv.GaussianBlur(image, (3, 3), 0)
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Gaussian blur', gaussian)
    cv.waitKey()
    cv.destroyAllWindows()

#Blurruing an image using Median Blur
def BLUR_Mead(image_path):
    # Reading the image
    image = cv.imread(image_path)
    # Applying the filter
    medianBlur = cv.medianBlur(image, 9)
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Median blur',medianBlur)
    cv.waitKey()
    cv.destroyAllWindows()

#Blurruing an image using Bilateral Blur
def BLUR_Bilat(image_path):
    # Reading the image
    image = cv.imread(image_path)
    # Applying the filter
    bilateral = cv.bilateralFilter(image,
    9, 75, 75)
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Bilateral blur', bilateral)
    cv.waitKey()
    cv.destroyAllWindows()

