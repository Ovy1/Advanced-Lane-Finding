### Advanced Lane Finding

In this project, our goal is to write a software pipeline to identify the lane boundaries in a video. Following steps were implemented to acheive the goal :

    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    Apply a distortion correction to raw images.
    Use color transforms, gradients, etc., to create a thresholded binary image.
    Apply a perspective transform to rectify binary image ("birds-eye view").
    Detect lane pixels and fit to find the lane boundary.
    Determine the curvature of the lane and vehicle position with respect to center.
    Warp the detected lane boundaries back onto the original image.
    Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The project consists of following files :


    Advanced-Lane-Finding-Submit.ipynb: IPython notebook with step-by-step description and execution of entire code.
    output images: output_images
    output video: result.mp4

--
#### Camera calibration
 I start by reading all the chess board images using calibration.add_image() function. The "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world, is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, calibration.obj_points is appended with a copy of the same coordinates every time I successfully detect all chessboard corners in a test image using the function calibration.find_corners(). calibration.img_points will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 
 
#### The algorithm

The algorithm is divided into two steps, in the first step we apply a perspective transform and compute a lane mask to identify potential locations of lane in an image, and in the next step we combine the lane mask information with previous frame information to compute the final lane. The second step is performed to discard effects of noisy or

##### Part 1: Get lane mask

Figure below presents the steps involved in obtaining lane masks from the original image. The steps are divided as follows,

1. Read and undistort image: In this step, a new image is read by the program and the image is undistorted using precomputed camera distortion matrices.
2. Perspective transform: Read in new image and apply perspective transform. Perspective transformation gives us bird's eye view of the road, this makes further processing easier as any irrelevant information about background is removed from the warped image.
3. Color Masks: Once we obtain the perspective transform, we next apply color masks to identify yellow and white pixels in the image. Color masks are applied after converting the image from RGB to HSV space. HSV space is more suited for identifying colors as it segements the colors into the color them selves (Hue), the ammount of color (Saturation) and brightness (Value). We identify yellow color as the pixels whose HSV-transformed intensities are between \([ 0, 100, 100]\) and \([ 50, 255, 255]\), and white color as the pixels with intensities between \( [20, 0, 180]\) and \([255, 80, 255] \).
4. Sobel Filters: In addition to the color masks, we apply sobel filters to detect edges. We apply sobel filters on L and S channels of image, as these were found to be robust to color and lighting variations. After multiple trial and error, we decided to use the magnitude of gradient along x- and y- directions with thresholds of 50 to 200 as good candidates to identify the edges.
5. Combine sobel and color masks: In a final step we combined candidate lane pixels from Sobel filters and color masks to obtain potential lane regions.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

One of the major step was in Identifying peaks in a histogram of the image to determine
location of lane lines then slicing the image in increments of 90 pixels, and using np.
Then identifying all non zero pixels around histogram peaks using the numpy function numpy.nonzero() and finally fitting a polynomial to each lane using the numpy.polyfit().

The curvature of the lines is also calculated in radius_of_curvature() function, With the inputs
being the x and y values of the left and right lines respectively, new lines are constructed using
np.polyfit(), using the pixel to meters transformation values xm_per_pix and ym_per_pix to get the lines in the units of meters. The curvature is then measured at the maximum value of the y
axis (at the bottom of the image) using the radius of a curve formula.

In the method Findline(), code exists to figure out frames where no lines could be found by the
sliding window search, or if the line gap was an outlier - which prompted a throw-away
frame(see Note after this section). In such scenarios we fall back to the previous frame for now


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the
lane and the position of the vehicle with respect to center.
The algorithm to calculate the radius of curvature can be found in the radius_of_curvature()
function here where the average or peaks of each line (left and right) are identified, slicing the
image in increments of 90 pixels, finally using the collection of x and y points we use np.polyfit
for a 2 degree line. The lines are then the output of the function.



##### Discussion
#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Over all my pipeline did a fairly good job of detecting the lane lines in the test video provided for the project, which shows a road in basically ideal conditions, with fairly distinct lane lines, and on
a clear day. However, it performs just ok on the challenge video when there are white segments,
The algorithm relies on having some sort of difference between the colors or shapes in the road,
thus if the road has no lanes, this algorithm would definitely fail.There were many trial-error-test-repeat cycles throughout this project. I would also like to explore
some tricks we learned in the first project and try to see if we can benefit from them. I would like
to solve some processing issues with canny_edge and see how they compare.
Resources that were helpeful during the project:
•
•
•
https://discussions.udacity.com/t/pixel-space-to-meter-space-conversion/241646
https://discussions.udacity.com/t/any-information-to-check-radius-of-curvature-and-off-
of-center/241681/2
https://discussions.udacity.com/t/trying-to-understand-sanity-check-validation-criteria-