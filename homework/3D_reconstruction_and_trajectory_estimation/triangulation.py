#code to do the 3d reconstruction and trajectory estimation assignment
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

def get_projection_matrices(K1, K2, R, T):
    # Compute the projection matrices for the two cameras
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(3, 1)))
    return P1, P2

def triangulate_points(P1, P2, points1, points2):
    # Triangulate the 3D points from the corresponding 2D points in the two images
    points1 = points1.reshape(-1,1,2)
    points2 = points2.reshape(-1,1,2)
    
    points4D = cv.triangulatePoints(P1, P2, points1, points2)
    points_3d = points4D[:3] / points4D[3]
    return points_3d.T

if __name__ == "__main__":
    points_left = cv.undistortPoints