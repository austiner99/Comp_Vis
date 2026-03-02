# code to calibrate the left and right cameras individually, images in file "pics" and in 3 folders 
# Pics/Left, Pics/Right, Pics/Both. Within each is a L and R folder with 50 images in either L or R or both, respectively.

'''
• Use the captured left and right image sequences to calibrate the left and right cameras individually.
• Find chessboard corners (subpixel) and do cameraCalibrate() for each camera.
• This task is the same as Task 2 in Assignment 2 but with different images.
• You can reuse the code and don’t have to submit the code again.
• Submit one set of the intrinsic (3×3) and distortion (5×1) parameters for each camera in your PDF File.
• You can and should download the practice images and the resulting calibration parameters from Learning Suit to confirm that
your code works. Please note that the stereo system used to capture these images is different (lens focal length, baseline, and
chessboard size) from the system for the baseball catcher. Read the instruction carefully.
'''

import numpy as np
import os
import cv2 as cv
import glob


# Task 1: Calibrate the left and right cameras individually using the chessboard images provided in the "pics" folder.
rows = 7
cols = 10

def calibrate_camera(image_folder, file):
    
    if file == 'Practice':
        square_size = 2.0 # inches
    elif file == 'Pics':
        square_size = 3.985 #inches
    else:
        raise ValueError("Invalid file name. Use 'Practice' or 'Pics'.")

    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] *= square_size # Multiply both x and y dimensions by square_size
    
    objpoints = []
    imgpoints = []
    
    if file == 'Practice':
        images = glob.glob(os.path.join(image_folder, '*.bmp'))
    elif file == 'Pics':
        images = glob.glob(os.path.join(image_folder, '*.png'))
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                        criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
    
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print(f"Camera matrix (K) for {image_folder}:\n{K}")
    print(f"Distortion coefficients (dist) for {image_folder}:\n{dist}")
    
    return K, dist, gray.shape[::-1]

# Task 2: Calibrate the stereo system using the chessboard images in the "Both" folder.

def stereo_calibrate(left_folder, right_folder, K1, dist1, K2, dist2, file):
    if file == 'Practice':
        square_size = 1.0 # inches
    elif file == 'Pics':
        square_size = 3.985 #inches
    else:
        raise ValueError("Invalid file name. Use 'Practice' or 'Pics'.")

    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] *= square_size  # Multiply both x and y dimensions by square_size
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    
    if file == 'Practice':
        left_images = glob.glob(os.path.join(left_folder, '*.bmp'))
        right_images = glob.glob(os.path.join(right_folder, '*.bmp'))
    elif file == 'Pics':
        left_images = glob.glob(os.path.join(left_folder, '*.png'))
        right_images = glob.glob(os.path.join(right_folder, '*.png'))
    
    for left_fname, right_fname in zip(left_images, right_images):
        img_left = cv.imread(left_fname)
        img_right = cv.imread(right_fname)
        
        gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)
        
        ret_left, corners_left = cv.findChessboardCorners(gray_left, (cols, rows), None)
        ret_right, corners_right = cv.findChessboardCorners(gray_right, (cols, rows), None)
        
        if ret_left and ret_right:
            objpoints.append(objp)
            corners2_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), 
                                            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners2_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), 
                                             criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints_left.append(corners2_left)
            imgpoints_right.append(corners2_right)
    
    ret_stereo, K1, dist1, K2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K1,
        dist1,
        K2,
        dist2,
        gray_left.shape[::-1],
        criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        flags=cv.CALIB_FIX_INTRINSIC
    )
    
    print(f"Rotation matrix (R):\n{R}")
    print(f"Translation vector (T):\n{T}")
    print(f"Essential matrix (E):\n{E}")
    print(f"Fundamental matrix (F):\n{F}")
    
    #convert R to rotation Vector (degrees)
    rvec, _ = cv.Rodrigues(R)
    rvec_degrees = np.degrees(rvec)
    print(f"Rotation vector (degrees):\n{rvec_degrees}")
    
    return R, T, E, F

#task 3: Epipolar Lines - use undistort to undistort lens distortion for both images
def draw_epipolar_lines(img_left, img_right, K1, dist1, K2, dist2, F):
    img_left_no_lines_undistorted = cv.undistort(img_left, K1, dist1)
    img_right_no_lines_undistorted = cv.undistort(img_right, K2, dist2)
    img_left_undistorted = img_left_no_lines_undistorted.copy()
    img_right_undistorted = img_right_no_lines_undistorted.copy()
    points_left = np.array([
        [200, 150],
        [300, 200],
        [400, 250],
        [250, 300]
    ], dtype=np.float32)
    points_right = np.array([
        [150, 200],
        [350, 220],
        [450, 260],
        [300, 320]
    ], dtype=np.float32) #MAKE THESE DIFFERENT FROM LEFT POINTS
    
    #draw circles on left and right images
    for point in points_left:
        cv.circle(img_left_undistorted, tuple(point.astype(int)), 6, (0,255,0), -1)

    for point in points_right:
        cv.circle(img_right_undistorted, tuple(point.astype(int)), 6, (0,255,0), -1)
    
    lines_right = cv.computeCorrespondEpilines(points_left.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines_left = cv.computeCorrespondEpilines(points_right.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    
    h_left, w_left = img_left_undistorted.shape[:2]
    h_right, w_right = img_right_undistorted.shape[:2]
    
    for r in lines_right:
        a, b, c = r
        x0, y0 = 0, int(-c / b)
        x1, y1 = w_right, int(-(c + a * w_right) / b)
        cv.line(img_right_undistorted, (x0, y0), (x1, y1), (255, 0, 0), 2)
    for l in lines_left:
        a, b, c = l
        x0, y0 = 0, int(-c / b)
        x1, y1 = w_left, int(-(c + a * w_left) / b)
        cv.line(img_left_undistorted, (x0, y0), (x1, y1), (255, 0, 0), 2)
    
    cv.imshow('Left Image with Epipolar Lines', img_left_undistorted)
    cv.imshow('Right Image with Epipolar Lines', img_right_undistorted)
    cv.imwrite('Left_Image_with_Epipolar_Lines.png', img_left_undistorted)
    cv.imwrite('Right_Image_with_Epipolar_Lines.png', img_right_undistorted)
    
    return img_left_undistorted, img_right_undistorted, img_left_no_lines_undistorted, img_right_no_lines_undistorted

#task 4: rectification - use stereorectify and initundistortrectifymap and remap to rectify images
    # draw horizontal lines at y = 50,100,150, ... on both images to show rectification and confirm image row alignment
    # save rectified images as Rectified_Left.png and Rectified_Right.png also include an absolute diff images of the
    # rectified images to show alignment (Rectified_Diff.png)
    # calculate and include rectification rotation matrix and vector
    
def rectify_images(img_left, img_right, K1, dist1, K2, dist2, R, T, image_size):
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(K1, dist1, K2, dist2, image_size, R, T)
    
    map1_left_x, map1_left_y = cv.initUndistortRectifyMap(K1, dist1, R1, P1, image_size, cv.CV_32FC1)
    map1_right_x, map1_right_y = cv.initUndistortRectifyMap(K2, dist2, R2, P2, image_size, cv.CV_32FC1)
    
    rectified_left = cv.remap(img_left, map1_left_x, map1_left_y, cv.INTER_LINEAR)
    rectified_right = cv.remap(img_right, map1_right_x, map1_right_y, cv.INTER_LINEAR)
    
    diff_image_left = cv.absdiff(rectified_left, img_left)
    diff_image_right = cv.absdiff(rectified_right, img_right)
       
    for y in range(50, rectified_left.shape[0], 50):
        cv.line(rectified_left, (0, y), (rectified_left.shape[1], y), (0, 255, 0), 1)
        cv.line(rectified_right, (0, y), (rectified_right.shape[1], y), (0, 255, 0), 1)
    

    cv.imshow('Rectified Left Image', rectified_left)
    cv.imshow('Rectified Right Image', rectified_right)
    cv.imshow('Difference Image Left', diff_image_left)
    cv.imshow('Difference Image Right', diff_image_right)
    
    cv.imwrite('Rectified_Left.png', rectified_left)
    cv.imwrite('Rectified_Right.png', rectified_right)
    cv.imwrite('Rectified_Diff_Left.png', diff_image_left)
    cv.imwrite('Rectified_Diff_Right.png', diff_image_right)
    
    #find rectification rotation matrix and vector
    R_rectification_left = R1
    R_rectification_right = R2
    rvec_left, _ = cv.Rodrigues(R_rectification_left)
    rvec_right, _ = cv.Rodrigues(R_rectification_right)
    rvec_left_degrees = np.degrees(rvec_left)
    rvec_right_degrees = np.degrees(rvec_right)
    print(f"Rectification rotation vector (degrees) for left camera:\n{rvec_left_degrees}")
    print(f"Rectification rotation vector (degrees) for right camera:\n{rvec_right_degrees}")
    print(f"Rectification rotation matrix for left camera:\n{R_rectification_left}")
    print(f"Rectification rotation matrix for right camera:\n{R_rectification_right}")
    
    return rectified_left, rectified_right, diff_image_left, diff_image_right

if __name__ == "__main__":
    main_file_name = input("Enter the main file name (Practice or Pics): ")
    if main_file_name not in ['Practice', 'Pics']:
        raise ValueError("Invalid file name. Use 'Practice' or 'Pics'.")
    left_folder = f'{main_file_name}/Left/L'
    right_folder = f'{main_file_name}/Right/R'
    both_left_folder = f'{main_file_name}/Both/L'
    both_right_folder = f'{main_file_name}/Both/R'
    
    K1, dist1, image_size = calibrate_camera(left_folder, main_file_name)
    K2, dist2, _ = calibrate_camera(right_folder, main_file_name)
    
    R, T, E, F = stereo_calibrate(both_left_folder, both_right_folder, K1, dist1, K2, dist2, main_file_name)
    
    img_left = cv.imread(os.path.join(both_left_folder, os.listdir(both_left_folder)[5]))
    img_right = cv.imread(os.path.join(both_right_folder, os.listdir(both_right_folder)[5]))
    cv.imwrite('Original_Left.png', img_left)
    cv.imwrite('Original_Right.png', img_right)
    
    undistorted_left, undistorted_right, img_left_no_lines_undistorted, img_right_no_lines_undistorted = draw_epipolar_lines(img_left, img_right, K1, dist1, K2, dist2, F)
    
    rect_left, rect_right, diff_left, diff_right = rectify_images(img_left, img_right, K1, dist1, K2, dist2, R, T, image_size)