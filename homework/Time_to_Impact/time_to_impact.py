
import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def measure_object(image, debug = True):
    # edges = cv2.Canny(image, 50, 75)
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours) == 0:
    #     return None
    # largest = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(largest)
    # print(f"Measured object at (x={x}, y={y}, w={w}, h={h})")
    #use hughlines instead
    edges = cv2.Canny(image, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=200, maxLineGap=90)
    if lines is None:
        return None
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 15:  # Vertical line
            vertical_lines.append((x1, y1, x2, y2))
    if len(vertical_lines) < 2:
        return None
    vertical_lines.sort(key=lambda line: line[0])  # Sort by x-coordinate
    x1, y1, x2, y2 = vertical_lines[0]
    x3, y3, x4, y4 = vertical_lines[-1]
    x = min(x1, x2, x3, x4)
    w = max(x1, x2, x3, x4) - x
    if debug:
        #show image with bounding box for debugging
        debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.line(debug_image, (x, 0), (x, image.shape[0]), (255, 0, 0), 2)
        cv2.line(debug_image, (x + w, 0), (x + w, image.shape[0]), (255, 0, 0), 2)
        plt.figure(figsize=(8, 6))
        plt.imshow(debug_image)
        plt.title("Detected Object with Bounding Box (click to close)")
        plt.axis('off')
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()
    return w

def compute_tti(points_1, points_2):
    dist = []
    for i in range(len(points_1)):
        for j in range(i + 1, len(points_1)):
            d1 = np.linalg.norm(points_1[i] - points_1[j])
            d2 = np.linalg.norm(points_2[i] - points_2[j])
            ds = d2 - d1
            if abs(ds) > 1e-6:  # Avoid division by zero
                if d1/ds > 0 and d1/ds < 1000:  # Filter out unrealistic TTI values
                    dist.append(d1 / ds)
    if len(dist) == 0:
        return None # Return None if no valid TTI values are found
    print(dist)
    print(len(dist))
    print(f"Mean TTI: {np.mean(dist)}")
    return np.mean(dist)

if __name__ == "__main__":
    time_to_impact_path = os.path.join(os.getcwd(), "Time_to_Impact_Images")
    filenames = sorted(glob.glob(os.path.join(time_to_impact_path, "*.jpg")), 
                      key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    
    # Task 0: Initialization of parameters and images
    K = np.array([[825.0900600547,    0.0000000000,  331.6538103208],
                  [0.0000000000,  824.2672147458,  252.9284287373],
                  [0.0000000000,    0.0000000000,    1.0000000000]])
    dist = np.array([  -0.2380769337, 
                     0.0931325835,  
                     0.0003242537, 
                     -0.0021901930,   
                     0.4641735616])
    
    can_width_mm = 59.0
    camera_motion_mm = 15.25
    fx = K[0, 0]
    
    converted_images = []
    #convert images to 8 bit
    for filename in filenames:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_8bit = cv2.convertScaleAbs(gray)
        converted_images.append(gray_8bit)
    
    # feature tracking - Start of Task 1 setup
    feature_params = dict(maxCorners=40, qualityLevel=0.01, minDistance=20, blockSize=7)
    lk_params = dict(winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    #detect initial features in the first frame
    p0 = cv2.goodFeaturesToTrack(converted_images[0], mask=None, **feature_params)
    
    tti_values = []
    distances_2 = []
    distances_3 = []
    debug = False
    
    prev = converted_images[0]
    
    #process frames for task 1
    for i in range(1, len(converted_images)):
        current = converted_images[i]
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, current, p0, None, **lk_params)
        
        # Filter out valid points
        good_old = p0[st == 1]
        good_new = p1[st == 1]
        if debug:
            debug_image = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
            for (x_old, y_old), (x_new, y_new) in zip(good_old, good_new):
                cv2.circle(debug_image, (int(x_new), int(y_new)), 5, (0, 255, 0), -1)
                cv2.line(debug_image, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (255, 0, 0), 2)
            plt.figure(figsize=(8, 6))
            plt.imshow(debug_image)
            plt.title(f"Frame {i} with Tracked Features (click to close)")
            plt.axis('off')
            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.close()
        
        tti = compute_tti(good_old, good_new)
        if tti is not None:
            tti_values.append(tti)
        else:
            tti_values.append(None)
        
        # Task 2: Known Object Velocity
        if tti is not None:
            distance_2 = camera_motion_mm * tti
            distances_2.append(distance_2)
        else:
            distances_2.append(None)
            
        # Task 3: Known Object Size and Camera Parameters
        # abs_diff_image = cv2.absdiff(prev, current)
        w = measure_object(current, debug=False)
        if w is not None:
            distance_3 = (can_width_mm * fx) / w
            distances_3.append(distance_3)
        else:
            distances_3.append(None)
        
        prev = current.copy()
        p0 = good_new.reshape(-1, 1, 2)
        
    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(tti_values, marker='o')
    plt.title('Time to Impact (Frames)')
    plt.xlabel('Frame Number')
    plt.ylabel('Time to Impact (Frames)')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(distances_2, marker='o', label='Known Velocity')
    plt.plot(distances_3, marker='x', label='Known Size')
    plt.title('Distance to Impact (mm)')
    plt.xlabel('Frame Number')
    plt.ylabel('Distance to Impact (mm)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plt.savefig("time_to_impact_results.png")
    plt.show()