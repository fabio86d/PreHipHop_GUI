"""
    Fabio D'Isidoro - ETH Zurich - February 2018

    A GUI for preprocessing of image registration:
    - Drawing of a mask in order to select specific areas in the image
    - Extracting edges with interactive Canny edge detection
    - Fitting a circle to the acetabluar cup with interactive Hough transform

    Based on opencv.

"""

# PYTHON MODULES
import os
import sys
import cv2
import numpy as np


# MY MODULES
import ROI_module as ROI


# LOOK UP TABLE: FASTER CONTRAST STRETCHING METHOD (from 16bit to 8bit ONLY)
def clip_and_rescale(img, min, max):

    image = np.array(img, copy = True) # just create a copy of the array
    image.clip(min,max, out = image)
    image -= min
    #image //= (max - min + 1)/256.
    image = np.divide(image,(max - min + 1)/256.)
    return image.astype(np.uint8)

def look_up_table(image, min, max):

    lut = np.arange(2**16, dtype = 'uint16')  # lut = look up table
    lut = clip_and_rescale(lut, min, max)

    return np.take(lut, image)  # it s equivalent to lut[image] that is "fancy indexing"


# Canny Edge callback function
def canny_edge(x):

    """Callback fuction to find edge with the Canny detector and circles with the Hough transform. """

    global processed_img, img, window_name_Canny, pp, cimg, img_mser_zoomed, original_height, original_width, factor

    # Get current parameters
    minImgContrast = cv2.getTrackbarPos('Min',window_name_Canny)
    maxImgContrast = cv2.getTrackbarPos('Max',window_name_Canny)
    CannyThresh1 = cv2.getTrackbarPos('Canny_Thresh1',window_name_Canny)
    CannyThresh2 = cv2.getTrackbarPos('Canny_Thresh2',window_name_Canny)
    Aperture = cv2.getTrackbarPos('Aperture',window_name_Canny)
    HoughParam1 = cv2.getTrackbarPos('Hough1',window_name_Canny)
    HoughParam2 = cv2.getTrackbarPos('Hough1',window_name_Canny)
    HoughParamDp = cv2.getTrackbarPos('HoughDp',window_name_Canny)
    HoughParamMinDist = cv2.getTrackbarPos('HoughMinDist',window_name_Canny)
    HoughParamMaxRadius = cv2.getTrackbarPos('HoughMaxRadius',window_name_Canny)
    HoughParamMinRadius = cv2.getTrackbarPos('HoughMinRadius',window_name_Canny)
    s = cv2.getTrackbarPos(switch,window_name_Canny)

    # Rescale 16bit into 8bit with contrast stretching
    rescaled_img = look_up_table(img, minImgContrast, maxImgContrast)

    if s == 0 and Aperture != 0:

        # Apply canny edge to current image
        processed_img = cv2.Canny(rescaled_img.astype(np.uint8), CannyThresh1, CannyThresh2,L2gradient=False,apertureSize= int(2*Aperture + 1)) # int(2*Aperture + 1)

    elif s == 1:

        # Apply canny edge to current image
        processed_img_ = cv2.Canny(rescaled_img.astype(np.uint8), CannyThresh1, CannyThresh2,L2gradient=False,apertureSize= int(2*Aperture + 1)) # int(2*Aperture + 1)

        # Resize current edge image
        zoomed_processed_img_ = cv2.resize(processed_img_[pp[0][1]: pp[1][1] ,pp[0][0]: (pp[0][0] + pp[1][1] - pp[0][1])], (int(original_height*factor), int(original_width*factor)), interpolation=cv2.INTER_LINEAR )

        # Find Hough circle from current zoomed edge image
        circles = cv2.HoughCircles(zoomed_processed_img_,cv2.HOUGH_GRADIENT,HoughParamDp,HoughParamMinDist,
                            param1=HoughParam1,param2=HoughParam2,minRadius=HoughParamMinRadius,maxRadius=HoughParamMaxRadius)

        cimg_zoomed = cv2.cvtColor(img_mser_zoomed,cv2.COLOR_GRAY2BGR)

        # Draw circles   
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,0:10]:
                # draw the outer circle
                cv2.circle(cimg_zoomed,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(cimg_zoomed,(i[0],i[1]),2,(0,0,255),3)

        # Display rescaled image
        processed_img = cimg_zoomed



##############################################################################################
# Load image
file_dir = sys.argv[1]
img = cv2.imread( file_dir, cv2.IMREAD_ANYDEPTH) # 16 bit image cv2.IMREAD_ANYDEPTH, 8 bit cv2.IMREAD_GRAYSCALE
processed_img = img.copy()
original_height, original_width = img.shape[:2]
factor = 1.0


# Define windows names
window_name_ROI = 'Draw ROIs (c = circle, r = rectangle, p = polygon. Press f to finish.'
window_name_Hough = 'Draw one single rectangle around cup. Press f to finish. '
window_name_Canny = 'GUI: Canny and Hough'


# Select ROI with ROI drawing tool (rectangles, circles, polygon can be drawn and composed together)
roi = ROI.ROI(img)
roi.save_masks_together('Composed_masks.tif')


## Select ROI for Hough detection
cv2.imshow(window_name_Hough,img)
roi_hough = ROI.ROIbuilder_rect(window_name_Hough,img)
pp = roi_hough.refPt_set[0]
img_mser_zoomed = cv2.resize(img[pp[0][1]: pp[1][1] ,pp[0][0]: (pp[0][0] + pp[1][1] - pp[0][1])], (int(original_height*factor), int(original_width*factor)), interpolation=cv2.INTER_LINEAR )

cv2.destroyAllWindows()

#Create Trackbars
cv2.namedWindow(window_name_Canny, cv2.WINDOW_NORMAL)
cv2.createTrackbar('Min',window_name_Canny,0,255,canny_edge)  # min Rescale Image
cv2.createTrackbar('Max',window_name_Canny,255,255,canny_edge)  # max Rescale Image
cv2.createTrackbar('Canny_Thresh1',window_name_Canny,0,256,canny_edge)  # Canny Edge Threshold 1
cv2.createTrackbar('Canny_Thresh2',window_name_Canny,0,256,canny_edge)  # Canny Edge Threshold 2
cv2.createTrackbar('Aperture',window_name_Canny,1,3,canny_edge)  # Canny Edge Threshold 2
cv2.createTrackbar('Hough1',window_name_Canny,20,200,canny_edge)  # Hough Transform param1
cv2.createTrackbar('Hough2',window_name_Canny,20,200,canny_edge)  # Hough Transform param2
cv2.createTrackbar('HoughDp',window_name_Canny,1,10,canny_edge)  # Hough Transform param2
cv2.createTrackbar('HoughMinDist',window_name_Canny,20,100,canny_edge)  # Hough Transform param2
cv2.createTrackbar('HoughMaxRadius',window_name_Canny,500,1000,canny_edge)  # Hough Transform param2
cv2.createTrackbar('HoughMinRadius',window_name_Canny,100,1000,canny_edge)  # Hough Transform param2
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, window_name_Canny,0,1,canny_edge)

while(1):

    cv2.imshow(window_name_Canny,processed_img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()