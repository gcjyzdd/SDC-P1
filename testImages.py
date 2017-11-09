# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    x_l = []
    y_l = []
    x_r = []
    y_r = []
    w_l = []
    w_r = []
    y_min_l = 539
    y_min_r = 539

    for line in lines:
        for x1, y1, x2, y2 in line:
            L = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            k = (y2 - y1)/(x2 - x1)

            if k>0:
                x_r.append((x1+x2)/2.0)
                y_r.append((y1+y2)/2.0)
                w_r.append(L)

                if y_min_r > y1:
                    y_min_r = y1

                if y_min_r > y2:
                    y_min_r = y2
            else:
                x_l.append((x1+x2)/2.0)
                y_l.append((y1+y2)/2.0)
                w_l.append(L)

                if y_min_l > y1:
                    y_min_l = y1

                if y_min_l > y2:
                    y_min_l = y2
            print( k, L)
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    X_l = np.array(x_l)
    X_l = np.vstack([X_l,np.ones(len(X_l))]).T
    Y_l = np.array(y_l)

    repeats_l = np.floor(w_l)
    X_l = np.repeat(X_l, list(repeats_l), axis=0)
    Y_l = np.repeat(Y_l, list(repeats_l), axis=0)

    m_l, c_l = np.linalg.lstsq(X_l, Y_l)[0]
    print(m_l,c_l)
    print('Done left')
    cv2.line(img, (math.floor((539-c_l)/m_l), 539), (math.floor((y_min_l-c_l)/m_l), y_min_l), color, thickness)

    X_r = np.array(x_r)
    X_r = np.vstack([X_r,np.ones(len(X_r))]).T
    Y_r = np.array(y_r)

    repeats_r = np.floor(w_r)
    X_r = np.repeat(X_r, list(repeats_r), axis=0)
    Y_r = np.repeat(Y_r, list(repeats_r), axis=0)

    m_r, c_r = np.linalg.lstsq(X_r, Y_r)[0]
    print(m_r,c_r)
    print('Done left')
    cv2.line(img, (math.floor((539-c_r)/m_r), 539), (math.floor((y_min_r-c_r)/m_r), y_min_r), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, color= [200, 0, 0], thickness= 4)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


print(os.listdir("./test_images"))

# reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

# printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)

plt.imshow(image)
plt.show()
# if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap = 'gray')

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
gray = grayscale(image)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 3*low_threshold
edges = canny(blur_gray, low_threshold, high_threshold)

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(60,imshape[0]),(409, 340), (544,340), (930,imshape[0])]], dtype=np.int32)
masked_img = region_of_interest(edges, vertices)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 5     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 8 #minimum number of pixels making up a line
max_line_gap = 5    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

lines = hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap)
output_im = weighted_img(lines,image)

plt.imshow(lines)
plt.show()

plt.imshow(output_im)
plt.show()

#import sys
#sys.exit()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 3*low_threshold
    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(50,imshape[0]),(405, 313), (535,313), (900,imshape[0])]], dtype=np.int32)
    masked_img = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 8 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    result = hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap)
    return weighted_img(result,image)


vid = 'solidWhiteRight.mp4'
#vid = 'solidYellowLeft.mp4'
#vid = 'challenge.mp4'
white_output = 'test_videos_output/' + vid
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/" + vid)
print( clip1.size)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
