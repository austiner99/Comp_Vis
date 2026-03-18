#code to do the 3d reconstruction and trajectory estimation assignment
import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt

def load_calibration_data(filename):
    data = np.load(filename)
    return data['K1'], data['dist1'], data['K2'], data['dist2'], data['R'], data['T']

def get_projection_matrices(K1, K2, R, T):
    # Compute the projection matrices for the two cameras
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))
    return P1, P2

# def get_projection_matrices_camera_2(K1, K2, R, T):
#     P1 = K1 @ np.hstack((R.T, -R.T @ T))
#     P2 = K2 @ np.hstack((np.eye(3), np.zeros((3, 1))))
#     return P1, P2

def triangulate_points(P1, P2, points1, points2):
    # Triangulate the 3D points from the corresponding 2D points in the two images
    points1 = points1.reshape(-1,1,2)
    points2 = points2.reshape(-1,1,2)
    
    points4D = cv.triangulatePoints(P1, P2, points1, points2)
    points_3d = points4D[:3] / points4D[3]
    return points_3d.T

# def verify_Pl_is_R_Pr_plus_T(points_1, points_2, R, T):
#     # Verify that the 3D points satisfy the relationship Pl = R * Pr + T
#     for i in range(points_1.shape[0]):
#         Pl = points_1[i]
#         Pr = points_2[i]
#         Pl_estimated = R @ Pr + T.flatten()
#         error = np.linalg.norm(Pl - Pl_estimated)
#         print(f"Point {i+1}: Pl = {np.round(Pl, 2)}, R*Pr + T = {np.round(Pl_estimated, 2)}, Error = {np.round(error, 2)}")

def show_baseball_with_markers(img, points):
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)
    return img

# def prove_rectified_translation(points_rectified, T):

#     print("\n===== Proving PL = PR + [||T||, 0, 0]^T =====\n")

#     baseline = np.linalg.norm(T)

#     for i, PL in enumerate(points_rectified):

#         PR = PL.copy()
#         PR[0] -= baseline

#         reconstructed_PL = PR + np.array([baseline, 0, 0])

#         error = np.linalg.norm(PL - reconstructed_PL)

#         print(f"Corner {i+1}:")
#         print("PL =", PL)
#         print("PR =", PR)
#         print("Reconstruction error =", error)
#         print()

#     # Distance check between two chessboard corners
#     dist = np.linalg.norm(points_rectified[0] - points_rectified[1])

#     print("Distance between two top corners:", dist)

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
    P1_right, P2_right = get_projection_matrices_camera_2(K1, K2, R, T)
    points_3d_right = triangulate_points(P1_right, P2_right, points_left_undistorted, points_right_undistorted)

    print("\n3D coordinates of the four outermost corners (left camera base):")
    for i, point in enumerate(points_3d):
        print(f"Corner {i+1}: {point}")
    
    print("\nPoints in right camera frame:")
    for i, point in enumerate(points_3d_right):
        print(f"Corner {i+1}: {point}")
    
    verify_Pl_is_R_Pr_plus_T(points_3d, points_3d_right, R, T)
    # Visualize original and undistorted points
    img_left_vis = img_left.copy()
    img_right_vis = img_right.copy()
    
    # Draw original points in green
    for point in points_left:
        x, y = int(point[0]), int(point[1])
        cv.circle(img_left_vis, (x, y), 5, (0, 255, 0), -1)
    
    for point in points_right:
        x, y = int(point[0]), int(point[1])
        cv.circle(img_right_vis, (x, y), 5, (0, 255, 0), -1)
    
    # Draw undistorted points in red
    for point in points_left_undistorted.reshape(-1, 2):
        x, y = int(point[0]), int(point[1])
        cv.circle(img_left_vis, (x, y), 5, (0, 0, 255), -1)
    
    for point in points_right_undistorted.reshape(-1, 2):
        x, y = int(point[0]), int(point[1])
        cv.circle(img_right_vis, (x, y), 5, (0, 0, 255), -1)
    
    cv.imshow('Left Image - Original (Green) vs Undistorted (Red)', img_left_vis)
    cv.imshow('Right Image - Original (Green) vs Undistorted (Red)', img_right_vis)
    cv.imwrite('left_image_comparison.png', img_left_vis)
    cv.imwrite('right_image_comparison.png', img_right_vis)
    cv.waitKey(0)
    cv.destroyAllWindows()

# def task2(K1, dist1, K2, dist2, R, T, left_img_path, right_img_path, board_size = (7,10)):
#     print("\nTask 2: Perspective Transorm")
#     image_left = cv.imread(left_img_path)
#     image_right = cv.imread(right_img_path)
#     gray_left = cv.cvtColor(image_left, cv.COLOR_BGR2GRAY)
#     gray_right = cv.cvtColor(image_right, cv.COLOR_BGR2GRAY)
#     ret_left, corners_left = cv.findChessboardCorners(gray_left, board_size)
#     ret_right, corners_right = cv.findChessboardCorners(gray_right, board_size)
#     if not ret_left or not ret_right:
#         print("Chessboard corners not found in one of the images.")
#         return
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
#     corners_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
#     corners_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
    
#     cols = board_size[0]
#     rows = board_size[1]
#     points_left = np.array([
#         corners_left[0], 
#         corners_left[cols-1], 
#         corners_left[-cols], 
#         corners_left[-1]]).reshape(-1, 2)
#     points_right = np.array([
#         corners_right[0], 
#         corners_right[cols-1], 
#         corners_right[-cols], 
#         corners_right[-1]]).reshape(-1, 2)
    
#     image_size = gray_left.shape[::-1]

#     # Stereo rectification
#     R1, R2, P1_rect, P2_rect, Q, _, _ = cv.stereoRectify(
#         K1, dist1, K2, dist2,
#         image_size, R, T,
#         flags=cv.CALIB_ZERO_DISPARITY
#     )

#     # Undistort + Rectify points
#     ptsL_rect = cv.undistortPoints(
#         points_left.reshape(-1,1,2),
#         K1, dist1,
#         R=R1,
#         P=P1_rect
#     )

#     ptsR_rect = cv.undistortPoints(
#         points_right.reshape(-1,1,2),
#         K2, dist2,
#         R=R2,
#         P=P2_rect
#     )

#     ptsL_rect = ptsL_rect.reshape(-1,2)
#     ptsR_rect = ptsR_rect.reshape(-1,2)

#     # Compute disparity
#     disparity = ptsL_rect[:,0] - ptsR_rect[:,0]

#     # Form (x,y,d) triplets
#     points_3D_input = np.zeros((4,1,3), dtype=np.float32)
#     points_3D_input[:,0,0] = ptsL_rect[:,0]
#     points_3D_input[:,0,1] = ptsL_rect[:,1]
#     points_3D_input[:,0,2] = disparity

#     # Apply perspective transform
#     points_3D_rect = cv.perspectiveTransform(points_3D_input, Q)

#     points_3D_rect = points_3D_rect.reshape(-1,3)

#     print("3D Points in Rectified Frame:")
#     print(points_3D_rect)

#     return points_3D_rect    

def detect_ball(frame, prev_frame, prev_points, roi=None, debug=False):
    gray_prev = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    gray_curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(gray_prev, gray_curr)
    if debug:
        cv.imshow('Debug - Frame Difference', diff)
    diff = cv.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv.threshold(diff, 35, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best_candidate = None
    best_score = 0
    max_change = 175  # Maximum allowed change in position between frames
    area_counter = 0
    perimeter_counter = 0
    change_counter = 0
    radius_counter = 0
    # score_counter = 0
    for c in contours:
        area = cv.contourArea(c)
        if area > 600:
            # print(f"Contour area {area} out of range, skipping.")
            area_counter += 1
            continue
        # perimeter = cv.arcLength(c, True)
        # if perimeter == 0:
        #     # print("Contour perimeter is zero, skipping.")
        #     perimeter_counter += 1
        #     continue
        # circularity = 4 * np.pi * area / (perimeter ** 2)
        # if circularity < 0.6:
        #     print(f"Circularity {circularity:.2f} too low, skipping.")
        #     continue
        (x, y), radius = cv.minEnclosingCircle(c)
        if prev_points is not None:
            change = np.linalg.norm(np.array([x, y]) - prev_points)
            if change > max_change:
                # print(f"Position change {change:.2f} too large, skipping.")
                change_counter += 1
                continue
        # if radius < 2 or radius > 50:
        #     # print(f"Contour radius {radius:.2f} out of range, skipping.")
        #     radius_counter += 1
        #     continue
        score = area  # Use area as score since circularity is disabled
        if score > best_score:
            best_score = score
            best_candidate = (x, y)
    if best_candidate is None:
        if debug:
            print(f"No valid ball candidate found. Counters: area={area_counter}, change={change_counter}")
        return None
    cx, cy = best_candidate
    if debug:
        # print(f"Best candidate: (cx={cx}, cy={cy}), score={best_score}")
        #print baseball image with marker
        img_with_marker = show_baseball_with_markers(frame.copy(), [best_candidate])
        cv.imshow('Debug - Detected Ball', img_with_marker)
        cv.waitKey(100)  # Display the debug image for 500 ms
    return np.array([cx, cy], dtype=np.float32)
        
def process_baseball_sequence(K1, K2, dist1, dist2, R, T, left_folder, right_folder):
    # left_images = sorted(glob.glob(os.path.join(left_folder, "*.png")))
    # right_images = sorted(glob.glob(os.path.join(right_folder, "*.png")))

    P1, P2 = get_projection_matrices(K1, K2, R, T)

    trajectory = []

    img_left_prev = None
    img_right_prev = None
    points_left = None
    points_right = None

    for i in range(0,50):
        img_left = cv.imread(os.path.join(left_folder, f"{i}.png"))
        img_right = cv.imread(os.path.join(right_folder, f"{i}.png"))
        if img_left_prev is not None and img_right_prev is not None:
            point_left = detect_ball(img_left, img_left_prev, prev_points= points_left, debug=False)
            point_right = detect_ball(img_right, img_right_prev,prev_points = points_right, debug=False)

            if point_left is None or point_right is  None:
                print(f"Ball not detected in both images for {os.path.join(left_folder, f'{i}.png')} and {os.path.join(right_folder, f'{i}.png')}. Skipping frame.")
                continue

            point_left = cv.undistortPoints(point_left.reshape(-1,1,2), K1, dist1, P=K1)
            point_right = cv.undistortPoints(point_right.reshape(-1,1,2), K2, dist2, P=K2)

            trajectory_now = triangulate_points(P1, P2, point_left, point_right)
            trajectory.append(trajectory_now[0])
            points_left = point_left
            points_right = point_right
            
            # Print every 5 frames
            frame_num = len(trajectory)
            if frame_num % 5 == 0:
                img_with_marker_left = show_baseball_with_markers(img_left.copy(), [point_left.reshape(2)])
                img_with_marker_right = show_baseball_with_markers(img_right.copy(), [point_right.reshape(2)])
                # cv.imshow('Left Image with Detected Ball', img_with_marker_left)
                # cv.imshow('Right Image with Detected Ball', img_with_marker_right)
                cv.imwrite(f'left_with_ball_{frame_num}.png', img_with_marker_left)
                cv.imwrite(f'right_with_ball_{frame_num}.png', img_with_marker_right)
                print(f"Frame {frame_num}: 3D Position = {trajectory_now[0]}")
                cv.waitKey(0)  # Wait for a key press to proceed to the next frame
        else:
            img_left_prev = img_left.copy()
            img_right_prev = img_right.copy()

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

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 10))
    
    ax1.scatter(Z, X, color='blue', label='Trajectory Points')
    ax1.plot(Z_line, X_model, color='red', label='Fitted Curve')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('X')
    ax1.set_title('X vs Z')
    ax1.invert_xaxis()
    ax1.legend()
    
    ax2.scatter(Z, Y, color='blue', label='Trajectory Points')
    ax2.plot(Z_line, Y_model, color='red', label='Fitted Curve')
    ax2.set_xlabel('Z')
    ax2.set_ylabel('Y')
    ax2.set_title('Y vs Z')
    ax2.invert_xaxis()
    ax2.legend()
    
    # Set same y-axis scale for both plots
    y_min = min(X.min(), Y.min())
    y_max = max(X.max(), Y.max())
    ax1.set_ylim(40, -10)
    ax2.set_ylim(-10,-60)
    
    plt.tight_layout()
    plt.savefig('trajectory_plots.png')  # Save the figure as a PNG file
    plt.show()

    #estimate final landing piont when z = 0

    x_final = np.polyval(X_fit, 0)
    y_final = np.polyval(Y_fit, 0)

    print(f"\nEstimated landing point: ({x_final:.2f}, {y_final:.2f}, 0)")
    
    #plot the 3D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='blue', label='Trajectory Points')
    ax.plot(X_model, Y_model, Z_line, color='red', label='Fitted Curve')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory of the Baseball')
    
    # Set same scale for X and Y axes
    max_range = max(X.max() - X.min(), Y.max() - Y.min()) / 2
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    
    ax.legend()
    plt.savefig('trajectory_3d.png')  # Save the figure as a PNG file
    plt.show()

if __name__ == "__main__":

    K1, dist1, K2, dist2, R, T = load_calibration_data('pics_stereo_calibration.npz')
    dist1 = dist1.reshape(-1,1)
    dist2 = dist2.reshape(-1,1)

    # task1(K1, dist1, K2, dist2, R, T, 'pics/Both/L/30.png', 'pics/Both/R/30.png')
    # points_3d = task2(K1, dist1, K2, dist2, R, T, 'pics/Both/L/30.png', 'pics/Both/R/30.png')
    # prove_rectified_translation(points_3d, T)

    trajectory = process_baseball_sequence(K1, K2, dist1, dist2, R, T, 'Baseball/L', 'Baseball/R')
    visualize_trajectory(trajectory)