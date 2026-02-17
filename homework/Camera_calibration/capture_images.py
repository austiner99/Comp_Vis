#use opencv and the camera hooked up to the computer to capture images at intervals of x seconds
import cv2
import time
import os

output_dir = os.path.join('homework', 'Camera_calibration', 'My_Camera_Calibration_Images')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
capture_interval = 2 # seconds
num_images = 20
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print('Press "q" to quit capturing images early.')

last_capture_time = time.time()
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    cv2.imshow('Camera', frame)

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        img_filename = os.path.join(output_dir, f'img_{img_count+1:02d}.jpg')
        cv2.imwrite(img_filename, frame)
        print(f'Captured {img_filename}')
        img_count += 1
        last_capture_time = current_time

        if img_count >= num_images:
            print("Captured all images.")
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting early.")
        break

cap.release()
cv2.destroyAllWindows()