'''
Task 3 - distortion correction
• Read in your saved intrinsic and distortion parameters from file(s).
• Download the three test images (Far, Close, Turned).
• Use OpenCV function undistort() to correct the distortion of these three images.
• Use OpenCV function absdiff() to compute the absolute difference between the original and undistorted images.
• Include all three absolute difference images in your PDF file.
• Submit your code for this task.
'''

import cv2 as cv
import numpy as np
import os

# Load intrinsic and distortion parameters from file
data = np.load('camera_calibration_params.npz')
mtx = data['intrinsic_matrix']
dist = data['distortion_coefficients']

# List of test images in distortion_correction_images folder
test_images = ['Far.jpg', 'Close.jpg', 'Turn.jpg']

for img_name in test_images:
    img_path = os.path.join('homework', 'Camera_calibration', 'distortion_correction_images', img_name)
    img = cv.imread(img_path)

    # Undistort the image
    undistorted_img = cv.undistort(img, mtx, dist)

    # Compute absolute difference
    abs_diff = cv.absdiff(img, undistorted_img)

    # Save the absolute difference image
    output_path = os.path.join('homework', 'Camera_calibration', 'distortion_correction_images', f'abs_diff_{img_name}')
    cv.imwrite(output_path, abs_diff)
    print(f"Absolute difference image saved to {output_path}")
    cv.imwrite(os.path.join('homework', 'Camera_calibration', 'distortion_correction_images', f'undistorted_{img_name}'), undistorted_img)
    print(f"Undistorted image saved to {os.path.join('homework', 'Camera_calibration', 'distortion_correction_images', f'undistorted_{img_name}')}")