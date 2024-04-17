import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import threading

sobel_kernel_size = 3
sobel_min_threshold = 90
conf_val = 0.85

MODEL_WEIGHT_PATH = 'best_obb1.pt'
model = YOLO(MODEL_WEIGHT_PATH)
bbox_id_counter = 1
bridge = CvBridge()

def get_rotated_box_points(x, y, width, height, angle):
    rectangle = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                          [width / 2, height / 2], [-width / 2, height / 2]])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotated_rectangle = np.dot(rectangle, rotation_matrix) + np.array([x, y])
    return np.int0(rotated_rectangle)

def empty_detect(img):
    global model, bbox_id_counter
    results = model(img)
    for box, conf in zip(results[0].obb, results[0].obb.conf):
        class_id = int(box.cls[0].item())
        confidence = float(conf.item())
        if class_id == 0 and confidence >= conf_val:
            x, y, w, h, r = box.xywhr[0].tolist()
            return (bbox_id_counter, x, y, w, h, r)
    return None

overlay_image = None

def image_callback(msg):
    global bbox_id_counter, overlay_image
    bbox_id_counter = 1
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print("CvBridge Error:", e)
        return

    # Always use a copy of the original image for modifications
    canvas = cv_image.copy()

    overlay_image = canvas.copy()

    bbox_info = empty_detect(cv_image)
    if bbox_info:
        id, x, y, w, h, r = bbox_info
        points = get_rotated_box_points(x, y, w, h, -r)

        buffer = 20
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        x_min = max(x_min - buffer, 0)
        y_min = max(y_min - buffer, 0)
        x_max = min(x_max + buffer, cv_image.shape[1])
        y_max = min(y_max + buffer, cv_image.shape[0])
        cropped_image = cv_image[y_min:y_max, x_min:x_max]

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
        sobel = cv2.magnitude(sobelx, sobely)

        _, sobel_thresholded = cv2.threshold(sobel, sobel_min_threshold, 255, cv2.THRESH_BINARY)

        # cv2.imshow('Sobel Thresholded', sobel_thresholded)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Apply the green mask on detected lines in the original image within the bounding box
        mask = (sobel_thresholded > 0)
        canvas[y_min:y_max, x_min:x_max][mask] = [0, 255, 0]  # Green mask on detected line

        # cv2.imshow('Sobel Thresholded with green line', canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Find contours which correspond to the lines of the parking lot.
        contours, _ = cv2.findContours(sobel_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming the two longest contours correspond to the left and right lines of the parking spot.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # Calculate the x-coordinates of the vertical lines (left and right)
        x_left = min([cv2.boundingRect(cnt)[0] for cnt in contours])
        x_right = max([cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] for cnt in contours])

        # Calculate the x-coordinate for the middle line
        x_middle = (x_left + x_right) // 2
        
        x_middle_pixels = [] 

        # Draw the middle line on the original image and generate pixel vals
        height, width = canvas.shape[:2]
        for y in range(height):
            x_middle_pixels.append(canvas[y, x_middle])
        canvas[y, x_middle] = (0, 0, 255)  # draw the line

        for pixel in x_middle_pixels:
            print(pixel)

        
        cv2.imshow('Middle Line Overlay', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    try:
        modified_image_pub.publish(bridge.cv2_to_imgmsg(canvas, "bgr8"))
    except CvBridgeError as e:
        print("CvBridge Error during modified image publishing:", e)

if __name__ == '__main__':
    rospy.init_node('image_processor', anonymous=True)
    image_sub = rospy.Subscriber("/oak/right/image_raw", Image, image_callback)
    modified_image_pub = rospy.Publisher("/oak/right/modified_image", Image, queue_size=10)

    try:
        # Start the spinning in a separate thread so that we can perform other tasks in the main thread
        spin_thread = threading.Thread(target=rospy.spin)
        spin_thread.start()

        while not rospy.is_shutdown():
            if overlay_image is not None:
                # If the global image variable is not None, show it
                cv2.imshow('Middle Line Overlay', overlay_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        spin_thread.join()  # Wait for the spin thread to finish
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
