import cv2
import numpy as np
import glob
import os

#task 1: Camera Calibration - Corner Detection
'''
TO DO:
• Download Camera Calibration Images from Learning Suite. - AR 1-40 in homework/Camera_calibration/Calibration_images folder
• Write your code to read in one of those calibration images.
• The input image must first be converted to grayscale using cvtColor() function (CV_RGB2GRAY).
• Use OpenCV function findChessboardCorners() to find chessboard inner corners.
• Use OpenCV function cornerSubPix() to refine corner locations.
• Use OpenCV function drawChessboardCorners() to draw corners (convert grayscale back to color before drawing).
• Include one output image with corners circled in color in your PDF file.
• This task is only an intermediate stage of calibration procedure. You dont have to submit your code. 
'''

# image_number = input("Enter image number (1-40): ")
for i in range(1, 41):
    print(f"Processing image AR{i}.jpg")
    img = cv2.imread(os.path.join('homework', 'Camera_calibration', 'Calibration_images', f'AR{i}.jpg'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chessboard_size = (10,7)  # Number of inner corners per a chessboard row and column
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img = cv2.drawChessboardCorners(img_color, chessboard_size, corners2, ret)
        output_path = os.path.join('homework', 'Camera_calibration', 'Calibration_corners', f'corners_AR{i}.png')
        cv2.imwrite(output_path, img_color)
        print(f"Corners detected and image saved to {output_path}")