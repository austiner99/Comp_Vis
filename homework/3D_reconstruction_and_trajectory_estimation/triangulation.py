#code to do the 3d reconstruction and trajectory estimation assignment
import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt

def load_calibration_data(filename):
    data = np.load(filename)
    return data['K1'], data['K2'], data['dist1'], data['dist2'], data['R'], data['T']

def get_projection_matrices(K1, K2, R, T):
    # Compute the projection matrices for the two cameras
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))
    return P1, P2

def triangulate_points(P1, P2, points1, points2):
    # Triangulate the 3D points from the corresponding 2D points in the two images
    points1 = points1.reshape(-1,1,2)
    points2 = points2.reshape(-1,1,2)
    
    points4D = cv.triangulatePoints(P1, P2, points1, points2)
    points_3d = points4D[:3] / points4D[3]
    return points_3d.T

def task1(K1, dist1, K2, dist2, R, T, left_img_path, right_img_path, chessboard_size=(7,10)):
    
    img_left = cv.imread(left_img_path)
    img_right = cv.imread(right_img_path)

    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    ret_left, corners_left = cv.findChessboardCorners(gray_left, chessboard_size)
    ret_right, corners_right = cv.findChessboardCorners(gray_right, chessboard_size)

    if not ret_left or not ret_right:
        print("Chessboard corners not found in one of the images.")
        return

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    corners_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

    cols = chessboard_size[0]
    rows = chessboard_size[1]

    #find four outermost points
    points_left = np.array([
        corners_left[0], 
        corners_left[cols-1], 
        corners_left[-cols], 
        corners_left[-1]]).reshape(-1, 2)
    
    points_right = np.array([
        corners_right[0], 
        corners_right[cols-1], 
        corners_right[-cols], 
        corners_right[-1]]).reshape(-1, 2)
    
    #undistort points
    points_left_undistorted = cv.undistortPoints(points_left, K1, dist1, P=K1)
    points_right_undistorted = cv.undistortPoints(points_right, K2, dist2, P=K2)

    P1, P2 = get_projection_matrices(K1, K2, R, T)
    points_3d = triangulate_points(P1, P2, points_left_undistorted, points_right_undistorted)

    print("\n3D coordinates of the four outermost corners:")
    for i, point in enumerate(points_3d):
        print(f"Corner {i+1}: {point}")

def detect_ball(frame, roi=None):

    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    
    c = max(contours, key=cv.contourArea)
    (x, y), radius = cv.minEnclosingCircle(c)

    if radius > 3: #CAN ADJUST THIS THRESHOLD
        return np.array([x,y], dtype=np.float32)
    
    return None

def process_baseball_sequence(K1, K2, dist1, dist2, R, T, left_folder, right_folder):
    left_images = sorted(glob.glob(os.path.join(left_folder, "*.png")))
    right_images = sorted(glob.glob(os.path.join(right_folder, "*.png")))

    P1, P2 = get_projection_matrices(K1, K2, R, T)

    trajectory = []

    for image_left_path, image_right_path in zip(left_images, right_images):
        img_left = cv.imread(image_left_path)
        img_right = cv.imread(image_right_path)

        point_left = detect_ball(img_left)
        point_right = detect_ball(img_right)

        if point_left is None and point_right is  None:
            print(f"Ball not detected in both images for {image_left_path} and {image_right_path}. Skipping frame.")
            continue

        point_left = cv.undistortPoints(point_left.reshape(-1,1,2), K1, dist1, P=K1)
        point_right = cv.undistortPoints(point_right.reshape(-1,1,2), K2, dist2, P=K2)

        trajectory_now = triangulate_points(P1, P2, point_left, point_right)
        trajectory.append(trajectory_now[0])

    trajectory = np.array(trajectory)
    return trajectory

def visualize_trajectory(trajectory):
    X = trajectory[:, 0]
    Y = trajectory[:, 1]
    Z = trajectory[:, 2]

    X_fit = np.polyfit(Z, X, 1)
    Y_fit = np.polyfit(Z, Y, 2)

    Z_line = np.linspace(Z.min(), Z.max(), 250)

    X_model = np.polyval(X_fit, Z_line)
    Y_model = np.polyval(Y_fit, Z_line)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(Z, X, color='blue', label='Trajectory Points')
    ax1.plot(Z_line, X_model, color='red', label='Fitted Curve')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('X')
    ax1.set_title('X vs Z')
    ax1.legend()
    
    ax2.scatter(Z, Y, color='blue', label='Trajectory Points')
    ax2.plot(Z_line, Y_model, color='red', label='Fitted Curve')
    ax2.set_xlabel('Z')
    ax2.set_ylabel('Y')
    ax2.set_title('Y vs Z')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    #estimate final landing piont when z = 0

    x_final = np.polyval(X_fit, 0)
    y_final = np.polyval(Y_fit, 0)

    print(f"\nEstimated landing point: ({x_final:.2f}, {y_final:.2f}, 0)")

if __name__ == "__main__":

    K1, dist, K2, dist2, R, T = load_calibration_data('Pics_stereo_calibration.npz')

    task1(K1, dist, K2, dist2, R, T, 'Pics/left01.png', 'Pics/right01.png')

    trajectory = process_baseball_sequence(K1, K2, dist, dist2, R, T, 'Baseball/left', 'Baseball/right')
    visualize_trajectory(trajectory)