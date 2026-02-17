'''
• Download the “Object with Corners” image to see the known object. You don’t have to process this image. Data are provided.
• Download the data file DataPoints.txt that has 20 image points (for x and y in pixels) and 20 object points (for x, y, z in inches).
• Write a program to read in the image and object points and use your calibration parameters from Task 2.
• Use the C++ version solvePnP() function or C version cvFindExtrinsicCameraParams2() (not preferable) to estimate the object
pose (measured by the camera). Convert R and T to 3´3 and 3´1 matrix (5-point deduction if not in these formats).
• Include the output rotation in both (3×3) and (3×1) and translation in (3×1) in your PDF file.
• Submit your code for this task.
'''

import cv2 as cv
import numpy as np
import os

#read DataPoints.txt file
data_points_path = os.path.join('homework', 'Camera_calibration', 'Object_Pose_Estimation', 'DataPoints.txt')

image_points = []
object_points = []

with open(data_points_path, 'r') as file:
    for line in file:
        #lines 1-20 are the image points, lines 21-40 are the object points
        values = list(map(float, line.split()))
        if len(image_points) < 20:
            image_points.append([values[0], values[1]])
        else:
            object_points.append([values[0], values[1], values[2]])
            
image_points = np.array(image_points, dtype=np.float32)
object_points = np.array(object_points, dtype=np.float32)

#solve PnP to get rotation and translation vectors
data = np.load('camera_calibration_params.npz')
mtx = data['intrinsic_matrix']
dist = data['distortion_coefficients']

success, rvec, tvec = cv.solvePnP(object_points, image_points, mtx, dist)
if success:
    R, _ = cv.Rodrigues(rvec)
    print("Rotation Matrix (3x3):")
    print(R)
    print("\nRotation Vector (3x1):")
    print(rvec)
    print("\nTranslation Vector (3x1):")
    print(tvec)
else:
    print("Pose estimation failed.")