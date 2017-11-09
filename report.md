# **Finding Lane Lines on the Road** 

## Summary of the method used in the code


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Deescription of the pipeline

Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

  1. First, I converted the images to grayscale;

  2. Applied a gaussion blurring filter with kernel size of 5 to the grayscale image;

  3. Got edges of the filtered image using Canny detection;

  4. Select ROI of the edge image and got the masked edges;

  5. Generated the final output image using alpha blending (linearly weight
  
In order to draw a single line on the left and right lanes, I first categorized the line segments to left and right parts based on their slopes. Then I tried to use least squares to get a best fiited line for both parts. The code is [testImages.py](./testImages.py). However, the result of the test video is not stable because some outlier points.

To improve the stability of the method, I used slope and intersection of exsiting(detected) lines instead of solving least squares. I computed the weighted cost of each detected line based on their length and selected the slope and intersection which has minimum cost. The code is [testImages_v2.py](testImages_v2.py) and is adopted in [P1_My.ipynb](P1_My.ipynb). Then I selectd the lower endpoint at the bottom of the image and the minimum y coordinate as the upper endpoint. The output test videos are stable.

Run ipython to check the result.


### 2. Potential shortcomings

  1. Fixed ROI selection. For different cameras positions and orientations, the real ROI could be different;
  
  2. Only two lane lines are detected. In real life, there may be three or more lane lines are available;
  
  3. Canny detection could fail because of light reflections.



### 3. Suggest possible improvements to your pipeline

  1. Include the car's size of camera calibration data to calculate a proper ROI;
  
  2. Use k means to get 2 or 3 lines;
  
  3. Add a memory mechanism like Kalman filter to enhance reliability of the algorithm.

