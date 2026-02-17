'''
• Repeat Task 2 (including saving calibration parameters in a file) using your own camera.
• You can use your real-time acquisition code for Assignment 1 to capture images.
• Use the chessboard for Assignment 1 (blocks must be square) and your code for Task 2 above to calibrate your camera.
• Make sure to change the number of corners entered to the calibration function in your code for Task 2.
• Make sure the chessboard paper is on a planar surface.
• Include the intrinsic (3×3) and distortion (5×1) parameters in your PDF file. (5-point deduction if not in these formats)
• Submit your code for this task.
'''

'''
task 2 - Intrinsic Parameters
• Write a program to read in all 40 of the calibration images one at a time in a loop.
• In the loop, find chessboard corners for each input image.
• Arrange corner points in the format for calibrateCamera() function. You need to learn how to use vector and vector of vectors if
you use C++. A PDF file (Using Vectors and Mat.pdf) can be found in Course Material/Reference on Learning Suite.
• Use OpenCV function calibrateCamera() to calculate the intrinsic and distortion parameters.
• The camera spec sheet says that the pixel size is 7.4µm´7.4µm and the sensor size is 4.8mm´3.6mm (1/3² format). Calculate and
report the actual focal length in mm.
• Include the intrinsic (3×3) and distortion (5×1) parameters in your PDF file. (5-point deduction if not in these formats)
• Submit your code for this task.
'''

import cv2
import numpy as np
import os
import glob

# Prepare object points based on the known chessboard pattern
chessboard_size = (9, 7)  # Number of inner corners per a chessboard row and column
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Read all calibration images (from capture_images.py code)
images = glob.glob(os.path.join('homework', 'Camera_calibration', 'My_Camera_Calibration_Images', 'img_**.jpg'))

for frame in images:
    img = cv2.imread(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
    else:
        print(f"Chessboard corners not found in image: {frame}")
        
# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Intrinsic Matrix (3x3):")
print(mtx)
print("\nDistortion Coefficients (5x1):")
print(dist)

#save parameters
np.savez('my_camera_calibration_params.npz', intrinsic_matrix=mtx, distortion_coefficients=dist)

# Calculate and report the actual focal length in mm
pixel_size_mm = 0.0074  # Pixel size in mm
focal_length_x_pixels = mtx[0, 0]
focal_length_mm = focal_length_x_pixels * pixel_size_mm
print(f"\nFocal Length in mm: {focal_length_mm} mm")