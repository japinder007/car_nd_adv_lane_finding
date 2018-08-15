import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import argparse
import pickle
from PIL import Image
import os

DEFAULT_COLOR_THRESHOLD = (190, 255)
DEFAULT_ABS_THRESHOLD = (0, 255)
DEFAULT_MAG_THRESHOLD = (0, 255)
DEFAULT_DIR_THRESHOLD = (0, np.pi/6)

DEFAULT_COLOR_SOBEL_KERNEL = 3
DEFAULT_DIR_SOBEL_KERNEL = 3
DEFAULT_MAG_SOBEL_KERNEL = 3
# Notes on shape.
# img_file = 'camera_cal/calibration1.jpg'
# img.shape
#   (720, 1280, 3)
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# gray.shape
#   (720, 1280)
# 720 is the height, 1280 is the width of the image here, 3 - channels.


def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def get_object_image_points(img_file, shape=(9, 6)):
    """
    Returns (obj_points, img_points) tuple for a chess board image.

    :param img_file (str) - Name of the image file to callibrate.
    :param shape tuple(int, int) - Shape of the checker board.

    :return (status, obj_points, img_points).
        status is true if the findChessboardCorners call succeeds.
        obj_points is the list of 3d points in the real world.
        img_points is the list of 2d points in the image.
    """
    img = mpimg.imread(img_file)
    obj_p = np.zeros((shape[0] * shape[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)
    gray = convert_to_grayscale(img)
    ret, corners = cv2.findChessboardCorners(gray, shape, None)
    # obj_points - 3d points in real word
    # img_points - 2d points in image plane
    return ret, obj_p, corners


def callibrate_camera(file_names, shape=(9, 6)):
    """
    Callibrates the camera given file_names of distorted chess board images.

    :param file_names (list[str]) - files containing chess board images.
    :param shape ((int, int)) - (corners per row, corners per column)
    :return (ret, mtx, dist, rvecs, tvecs).
        The return value is the output of calibrateCamera
    """
    all_obj_points = []
    all_img_points = []
    prev_shape = None
    for f in file_names:
        print('Processing {f}'.format(f=f))
        status, obj_points, img_points = get_object_image_points(f, shape)
        curr_shape = convert_to_grayscale(mpimg.imread(f)).shape
        if prev_shape:
            if prev_shape != curr_shape:
                print('Shapes mismatch for {f} : {p} vs {c}'.format(
                    f=f, p=prev_shape, c=curr_shape))
            prev_shape = curr_shape
        if status:
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
        else:
            print('Could not find chess board corners for {f}'.format(f=f))
    return cv2.calibrateCamera(
        all_obj_points, all_img_points, curr_shape, None, None
    )


def undistort_and_save_image(output_dir, file_name, mtx, dist):
    print('Undistorting {f} {m} {d}'.format(f=file_name, m=mtx, d=dist))
    dist = cv2.undistort(mpimg.imread(file_name), mtx, dist, None, mtx)
    im = Image.fromarray(dist)
    base_name = os.path.basename(file_name)
    prefix = base_name.split('.')[0]
    full_path = os.path.join(output_dir, '{p}_dist.jpg'.format(p=prefix))
    print('Undistorting file {f} to {o}'.format(f=file_name, o=full_path))
    im.save(full_path)


def sobel_abs_threshold(img, orient='x', thresh=DEFAULT_ABS_THRESHOLD):
    """
    Return the mage after applying threshold on sobel.
    :param img (numpy array) - image.
    :param orient (str) - Either 'x' or 'y'.
    :param thresh_min (int) - Minimum threshold value.
    :param thresh_max (int) - Maximum threshold value.

    :return returns the image with only the pixels which have sobel values in
        the given range.
    """
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = convert_to_grayscale(img)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    result = np.zeros_like(scaled_sobel)
    result[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return result


def sobel_mag_threshold(img, sobel_kernel=DEFAULT_MAG_SOBEL_KERNEL, mag_thresh=DEFAULT_MAG_THRESHOLD):
    """
    Return the image after applying threshold on magnitude of sobel.
    :param img (numpy array) - image.
    :param orient (str) - Either 'x' or 'y'.
    :param thresh_min (int) - Minimum threshold value.
    :param thresh_max (int) - Maximum threshold value.

    :return returns the image with only the pixels which have sobel values in
        the given range.
    """
    # Convert to grayscale
    gray = convert_to_grayscale(img)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=DEFAULT_DIR_SOBEL_KERNEL, thresh=DEFAULT_DIR_THRESHOLD):
    # Grayscale
    gray = convert_to_grayscale(img)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def color_threshold(img, thresh=DEFAULT_COLOR_THRESHOLD):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def combined_gradient_color_threshold(
    img,
    grad_x_thresh=DEFAULT_ABS_THRESHOLD,
    color_thresh=DEFAULT_COLOR_THRESHOLD
):
    s_binary = color_threshold(img, color_thresh)
    sxbinary = sobel_abs_threshold(
        img, orient='x', thresh=grad_x_thresh
    )
    # Combine color and x gradient threshold results.
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def combined_dir_color_threshold(
    img,
    grad_x_thresh=DEFAULT_ABS_THRESHOLD,
    dir_thresh=DEFAULT_DIR_THRESHOLD
):
    s_binary = color_threshold(img, color_thresh)
    threshold_binary = dir_threshold(img, thresh=dir_thresh)
    combined_binary = np.zeros_like(threshold_binary)
    combined_binary[(s_binary == 1) | (threshold_binary == 1)] = 1
    return combined_binary


"""
dst_points=np.float32([
    [268.552, 10],
    [1034.03, 10],
    [268.552, 675.317],
    [1034.03, 675.317],
])
"""


def get_perspective_transform_matrix(
    mtx,
    dist,
    file_name='test_images/straight_lines1.jpg',
    src_points=np.float32([
        [576.04, 464.487], [707.079, 464.487],
        [268.552, 675.317], [1034.03, 675.317]]
    ),
    dst_points=np.float32([
        [150, 150],
        [1000, 150],
        [150, 700],
        [1000, 700],
    ])
):
    """
    Applies perspective transform to undist, mapping src_points to dst_points.
    This produces an image of the same size as the original image.

    :param mtx (numpy array) - Matrix for removing distortion.
    :param dist (numpy array) - Matrix for displacement.
    :param file_name (str) - Name of the image file.
    :param src_points (list[points]) - List of points in the source image.
        These are assumed to be four points provided in the following order -
        Top left, top right, bottom left, bottom right.
    :param dst_points (list[points]) - Points with 1:1 mapping to src_points.

    :return returns the matrix which produces the perspective transform.
    """
    img = mpimg.imread(file_name)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    gray = convert_to_grayscale(undist)
    # img_size is width x height :-(.
    width = gray.shape[1]
    height = gray.shape[0]
    return (cv2.getPerspectiveTransform(src_points, dst_points), cv2.getPerspectiveTransform(dst_points, src_points))


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Draws lines over the image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def get_points_for_lanes(src):
    return [
        [(src[0][0], src[0][1], src[2][0], src[2][1])],
        [(src[1][0], src[1][1], src[3][0], src[3][1])],
    ]


def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]
    # Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    return histogram


def find_lane_pixels(binary_warped):
    """
    Returns the left and right lane pixels.

    :param binary_warped - Warped image.

    Returns (leftx, lefty, rightx, righty, out_img) where
        leftx - x coordinates of the points in the left lane.
        rightx - y coordinates of the points in the left lane.
        rightx - x coordinates of the points in the right lane.
        righty - y coordinates of the points in the right lane.
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Visualization #
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fitx, right_fitx, left_fit, right_fit


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty, left_fit, right_fit


def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # TO-DO: Set the area of search based on activated x-values ###
    # within the +/- margin of our polynomial function ###
    # Hint: consider the window areas for the similarly named variables ###
    # in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                    right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit_next, right_fit_next = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    # Visualization #
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, left_fitx, right_fitx, left_fit_next, right_fit_next, leftx, lefty, rightx, righty


def get_curvature(fit, y):
    """
    Returns the curvature for a given fit and ploty values.

    :param fit (array) - Coefficients of second order polynomial fit to the lane.
    :param ploty (array) - Values of y along which to evaluate the curvature on.

    :return array - Values of curvature at the corresponding ploty values.
    """
    # ((1 + (2Ay + B) ^ 2)^3/2)/2A
    A = fit[0]
    B = fit[1]
    t1 = 2 * A * y * y + B
    t2 = np.square(t1)
    t3 = (1 + t2)
    t4 = np.power(t3, 1.5)
    t5 = t4 / np.absolute(2 * A)
    return t5


def draw_lane(
    original_img, binary_img, left_fit, right_fit, inv_perspective_trans
):
    new_img = np.copy(original_img)
    if left_fit is None or right_fit is None:
        return original_img

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h, w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Prepare data for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(
        color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 255),
        thickness=15
    )
    cv2.polylines(
        color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 255),
        thickness=15
    )

    newwarp = cv2.warpPerspective(color_warp, inv_perspective_trans, (w, h))
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result


def transform_image(img, mtx, dist, p_transform_matrix,
                    grad_x_thresh=DEFAULT_ABS_THRESHOLD,
                    color_threshold=DEFAULT_COLOR_THRESHOLD):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Now draw the warped image.
    height, width = img.shape[:2]
    p_transformed = cv2.warpPerspective(
        undist, p_transform_matrix, (width, height),
        flags=cv2.INTER_LINEAR
    )
    threshold_binary = combined_gradient_color_threshold(
        p_transformed, grad_x_thresh, color_threshold
    )

    out_img, left_fitx, right_fitx, left_fit, right_fit = fit_polynomial(threshold_binary)
    return undist, threshold_binary, p_transformed, out_img, left_fitx, right_fitx, left_fit, right_fit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p', '--pickle_file_name', default='camera_cal.pkl',
                        help='File to store mtx and dist into')
    parser.add_argument('-i', '--index', default=0, type=int,
                        help='Index into file_names')
    parser.add_argument('-t', '--test_image_name', type=str, help='Image to transform')
    args = parser.parse_args()
    pickle_file_name = args.pickle_file_name
    file_names = glob.glob('camera_cal/calibration*.jpg')
    try:
        with open(pickle_file_name, 'rb') as f:
            print('Restoring mtx and dist from {f}'.format(f=pickle_file_name))
            pickle_dict = pickle.load(f)
            mtx = pickle_dict['mtx']
            dist = pickle_dict['dist']
    except IOError as err:
        print('Generating mtx and dist and saving to {f}'.format(f=pickle_file_name))
        ret, mtx, dist, rvecs, tvecs = callibrate_camera(file_names)
        pickle.dump({'mtx': mtx, 'dist': dist}, open(pickle_file_name, 'wb'))

    # draw_lines(undist, [
    #    [(521.697, 500.501, 234.025, 699.821)],
    #    [(764.666, 500.501, 1064.43, 699.821)]
    # ])
    # draw_lines(undist, get_points_for_lanes(src_points))

    p_transform_matrix, p_transform_matrix_inv = get_perspective_transform_matrix(mtx, dist)
    img = mpimg.imread(args.test_image_name)
    undist, threshold_binary, p_transformed, out_img, left_fitx, right_fitx, left_fit, right_fit = transform_image(
        img, mtx=mtx, dist=dist,
        p_transform_matrix=p_transform_matrix,
        color_threshold=(150, 255),
        grad_x_thresh=(20, 255)
    )

    f, axs = plt.subplots(2, 2, figsize=(24, 10))
    [ax1, ax2, ax3, ax4] = plt.gcf().get_axes()
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(threshold_binary)
    ax2.set_title('Thresholded', fontsize=10)
    ax3.imshow(p_transformed)
    ax3.set_title('Perspective', fontsize=10)
    ax4.imshow(out_img)
    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
    ax4.plot(left_fitx, ploty, color='yellow')
    ax4.plot(right_fitx, ploty, color='yellow')
    ax4.set_title('Pipeline result', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #plt.imshow(out_img)
    plt.show()
