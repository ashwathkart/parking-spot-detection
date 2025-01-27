import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO

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

def image_callback(msg):
    global bbox_id_counter
    bbox_id_counter = 1
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print("CvBridge Error:", e)
        return

    # Always use a copy of the original image for modifications
    canvas = cv_image.copy()

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

        # Apply the green mask on detected lines in the original image within the bounding box
        mask = (sobel_thresholded > 0)
        canvas[y_min:y_max, x_min:x_max][mask] = [0, 255, 0]  # Green mask on detected line

    try:
        modified_image_pub.publish(bridge.cv2_to_imgmsg(canvas, "bgr8"))
    except CvBridgeError as e:
        print("CvBridge Error during modified image publishing:", e)

if __name__ == '__main__':
    rospy.init_node('image_processor', anonymous=True)
    image_sub = rospy.Subscriber("/oak/right/image_raw", Image, image_callback)
    modified_image_pub = rospy.Publisher("/oak/right/modified_image", Image, queue_size=10)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
