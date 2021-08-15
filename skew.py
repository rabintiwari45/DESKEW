### FIRST APPROACH CODE. SOLVED USING PROJECTION PROFILE METHOD.

## This approach works fine but can be very slow for high value 
## of MAX_LIMIT and small value of DELTA. If the image dimension
## big then it will be really slow. So, we approached the very same
## problem with Hough Transform method.

## Loading the necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter

## Path of file
FILENAME = 'PATH/TO/IMAGE'

## Initializing the variables
DELTA = 0.1
MAX_LIMIT = 45

## Function to convert an image to binary image
def binary_image(filename):
  image = im.open(filename)
  width,height = image.size
  binary_pixel = np.array(image.convert('1').getdata(), np.uint8)
  binary_img = 1 - (binary_pixel.reshape((height, width)) / 255.0)
  return binary_img

## Function to find the score(difference between peaks of histogram)
def find_score(array,angle):
  data = inter.rotate(array, angle, reshape=False, order=0)
  hist = np.sum(data, axis=1)
  score = np.sum((hist[1:] - hist[:-1]) ** 2)
  return hist, score

## Function to rotated an image
def rotate_image(image, angle):
    mean_pixel = np.median(np.median(image, axis=0), axis=0)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=mean_pixel)
    return result

def main():
  image = cv2.imread(FILENAME)
  binary_img = binary_image(FILENAME)
  angles = np.arange(-MAX_LIMIT, MAX_LIMIT+DELTA, DELTA)
  scores = []
  for angle in angles:
    hist, score = find_score(binary_img, angle)
    scores.append(score)
  best_score = max(scores)
  best_angle = angles[scores.index(best_score)]
  rotated_image = rotate_image(image,best_angle)
  cv2.imwrite('rotated_image.jpeg', rotated_image)

if __name__ == '__main__':
  main()
  

### SECONG APPROACH CODE. SOLVED USING HOUGH TRANSFORM.

## Loading the necessary libraries
import re
import cv2
import numpy as np
import pytesseract  ## to check the orientation
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import mode
from skimage import io
from skimage.filters import threshold_otsu, sobel
from PIL import Image

## Path of the file
FILE_NAME = 'PATH/TO/IMAGE'

## Function to binarize the image
def binarize_image(rgb_image):
  image = rgb2gray(rgb_image)
  threshold = threshold_otsu(image)
  binary_image = image < threshold
  return binary_image

## Function for edge detection
def find_edges(bina_image):
  image_edges = sobel(bina_image)
  return image_edges

## Function to calculate skew angle using Hough Transform
def skew_angle(image_edges):
  h, theta, d = hough_line(image_edges)
  accum, angles, dists = hough_line_peaks(h, theta, d)
  angle = mode(np.around(angles, decimals=2))[0]
  angle = np.rad2deg(mode(angles)[0][0])
  if (angle < 0):
    # rotating in anti clockwise direction
    skew_angle = angle + 90
  else:
    # rotating in clockwise direction
    skew_angle = angle - 90
  return skew_angle

## Function to rotate an image
def rotate_image(image, angle):
    mean_pixel = np.median(np.median(image, axis=0), axis=0)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=mean_pixel)
    return image

## Function to check the orientation of image
def orientated_image(image):
  try:
    newdata = pytesseract.image_to_osd(image)
    angle = int(re.search('(?<=Rotate: )\d+', newdata).group(0))
    if angle == 0:
      return ndimage.rotate(image,angle)
    else:
      rotated_image = ndimage.rotate(image,angle)
      data = pytesseract.image_to_osd(rotated_image)
      angle_next = int(re.search('(?<=Rotate: )\d+', data).group(0))
      if angle_next == 0:
        return ndimage.rotate(image,angle)
      else:
        return ndimage.rotate(rotated_image,angle_next)
  except:
    return ndimage.rotate(image,0)
    
def main():
  image = Image.open(FILE_NAME)
  ## Check the orientation of the image
  image_orie = orientated_image(image)
  ## Running the above function
  bin_image = binarize_image(image_orie)
  image_edges = find_edges(bin_image)
  angle = skew_angle(image_edges)
  rotated_image = rotate_image(image_orie,angle)
  cv2.imwrite('rotated_image.jpg', rotated_image)
  

if __name__ == '__main__':
  main()
