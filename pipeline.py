import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import argparse
import pickle
from PIL import Image
import os


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
    dist = cv2.undistort(mpimg.imread(file_name), mtx, dist, None, mtx)
    im = Image.fromarray(dist)
    base_name = os.path.basename(file_name)
    prefix = base_name.split('.')[0]
    full_path = os.path.join(output_dir, '{p}_dist.jpg'.format(p=prefix))
    print('Undistorting file {f} to {o}'.format(f=file_name, o=full_path))
    im.save(full_path)


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
            pickle_dict = pickle.load(f)
            mtx = pickle_dict['mtx']
            dist = pickle_dict['dist']
    except IOError as err:
        ret, mtx, dist, rvecs, tvecs = callibrate_camera(file_names)
        pickle.dump({'mtx': mtx, 'dist': dist}, open(pickle_file_name, 'wb'))

    for f in file_names:
        undistort_and_save_image('camera_cal_undist', f, mtx, dist)
