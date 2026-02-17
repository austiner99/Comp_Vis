import cv2 as cv
import numpy as np
import os

#use the left and right frames in Sequence1 to detect the ball being thrown 
def detect_baseball(prev_frame, curr_frame):
    # Convert frames to grayscale
    gray_prev = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    gray_curr = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

    diff = cv.absdiff(gray_prev, gray_curr)
    _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    c = max(contours, key=cv.contourArea)
    
    if cv.contourArea(c) < 4:
        return None
    
    (x,y),radius = cv.minEnclosingCircle(c)
    center = (int(x), int(y))
    
    return center, int(radius)

# Define the codec and create a VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('combined_output.avi', fourcc, 20.0, (640 * 2, 480))  # Adjust the frame size as needed

# Your existing detect_baseball function here...

if __name__ == "__main__":
    
    prev_left = None
    prev_right = None
    for i in range(40, 79):
        frame_num = i
        left_image_path = os.path.join('homework', 'Sequence1',f'L{frame_num}.png')
        right_image_path = os.path.join('homework', 'Sequence1',f'R{frame_num}.png')

        left_frame = cv.imread(left_image_path)
        right_frame = cv.imread(right_image_path)

        if prev_left is not None:
            result_left = detect_baseball(prev_left, left_frame)
            if result_left:
                center,radius = result_left
                cv.circle(left_frame, center, radius, (0, 255, 0), 2)
                cv.circle(left_frame, center, 3, (0, 0, 255), -1)
                
        if prev_right is not None:
            result_right = detect_baseball(prev_right, right_frame)
            if result_right:
                center,radius = result_right
                cv.circle(right_frame, center, radius, (0, 255, 0), 2)
                cv.circle(right_frame, center, 3, (0, 0, 255), -1)
        
        combined = np.hstack([left_frame, right_frame])
        cv.imshow('Left and Right Frames', combined)

        # Write the frame to the video file
        out.write(combined)

        prev_left = left_frame.copy()
        prev_right = right_frame.copy()
        
        if cv.waitKey(100) == 27:  # Press 'Esc' to exit
            break

    # Release the video writer and destroy all windows
    out.release()
    cv.destroyAllWindows()
