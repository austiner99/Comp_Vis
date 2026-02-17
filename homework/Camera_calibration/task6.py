'''
Task 6 - Distortion correction with your own camera
• Repeat Task 3 (including reading calibration parameters from a file) using your own camera.
• Include an image of the absolute difference between the original (captured from your camera) and the undistorted images.
• Submit your code for this task.
'''

import cv2 as cv
import numpy as np
import os

# Load intrinsic and distortion parameters from file
data = np.load('my_camera_calibration_params.npz')
mtx = data['intrinsic_matrix']
dist = data['distortion_coefficients']

# List of test images in distortion_correction_images folder
test_images = ['img_01.jpg', 'img_10.jpg', 'img_20.jpg']

for img_name in test_images:
    img_path = os.path.join('homework', 'Camera_calibration', 'My_Camera_Calibration_Images', img_name)
    img = cv.imread(img_path)

    # Undistort the image
    undistorted_img = cv.undistort(img, mtx, dist)

    # Compute absolute difference
    abs_diff = cv.absdiff(img, undistorted_img)

    # Save the absolute difference image
    output_path = os.path.join('homework', 'Camera_calibration', 'My_Camera_Images_Undistorted', f'abs_diff_{img_name}')
    cv.imwrite(output_path, abs_diff)
    print(f"Absolute difference image saved to {output_path}")
    cv.imwrite(os.path.join('homework', 'Camera_calibration', 'My_Camera_Images_Undistorted', f'undistorted_{img_name}'), undistorted_img)
    print(f"Undistorted image saved to {os.path.join('homework', 'Camera_calibration', 'My_Camera_Images_Undistorted', f'undistorted_{img_name}')}")