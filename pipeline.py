import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import argparse
import pickle
from PIL import Image
import os

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


def sobel_abs_threshold(img, orient='x', thresh_min=0, thresh_max=255):
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
    result[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return result


def sobel_mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
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


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
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


def color_threshold(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def combined_gradient_color_threshold(img, grad_x_thresh=(20, 100), color_thresh=(170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 1

    gray = convert_to_grayscale(img)
    # Take the derivative in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= grad_x_thresh[0]) & (scaled_sobel <= grad_x_thresh[1])] = 1

    # Combine color and x gradient threshold results.
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def perspective_transform(undist, src_points, dst_points, img_shape):
    """
    Applies perspective transform to undist, mapping src_points to dst_points.
    This produces an image of the same size as the original image.

    :param undist (numpy array) - Undistorted image.
    :param src_points (list[points]) - List of points in the source image.
        These are assumed to be four points provided in the following order -
        Top left, top right, bottom left, bottom right.
    :return returns the image after applying the perspective transform.
    """
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(undist, M, img_shape)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p', '--pickle_file_name', default='camera_cal.pkl',
                        help='File to store mtx and dist into')
    parser.add_argument('-i', '--index', default=0, type=int,
                        help='Index into file_names')
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

    # for f in file_names:
    #    undistort_and_save_image('camera_cal_undist', f, mtx, dist)
    img = mpimg.imread('test_images/straight_lines1.jpg')
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    src_points = np.float32([
        [521.697, 500.501],
        [764.666, 500.501],
        [234.025, 699.821],
        [1064.43, 699.821]
    ])
    src_points = np.float32([
        [596.376, 450.54],
        [677.695, 450.09],
        [234.025, 699.821],
        [1064.43, 699.821]
    ])
    src_points = np.float32([
        [560.614, 475.258],
        [723.645, 475.258],
        [234.025, 699.821],
        [1068.54, 699.821]
    ])
    src_points = np.float32([
        [576.04, 464.487],
        [707.079, 464.487],
        [268.552, 675.317],
        [1034.03, 675.317]
    ])
    # draw_lines(undist, [
    #    [(521.697, 500.501, 234.025, 699.821)],
    #    [(764.666, 500.501, 1064.43, 699.821)]
    # ])
    draw_lines(undist, get_points_for_lanes(src_points))

    # Compute the matrix to transform src_points to dst_points. The first point
    # is the axis along the width and the second is along the height.
    offset = 10
    gray = convert_to_grayscale(undist)
    # img_size is width x height :-(.
    width = gray.shape[1]
    height = gray.shape[0]
    dst_points = np.float32([
        [offset, offset],
        [width - offset, offset],
        [offset, height - offset],
        [width - offset, height - offset],
    ])
    dst_points = np.float32([
        [268.552, offset],
        [1034.03, offset],
        [268.552, 675.317],
        [1034.03, 675.317],
    ])
    p_transformed = perspective_transform(undist, src_points, dst_points, img_shape=(width, height))
    # undistort_and_save_image(
    #    'test_images_undist_perspective',
    #    'test_images/straight_lines1.jpg', mtx, dist
    # )
    # threshold_binary = combined_gradient_color_threshold(undist)
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(undist)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(p_transformed)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
