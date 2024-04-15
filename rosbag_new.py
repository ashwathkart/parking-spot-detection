import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO

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
        if class_id == 0 and confidence >= 0.70:  # Assuming '0' is the class for empty spots
            x, y, w, h, r = box.xywhr[0].tolist()
            return (bbox_id_counter, x, y, w, h, r)
    return None

def image_callback(msg):
    global bbox_id_counter
    bbox_id_counter = 1
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)

    bbox_info = empty_detect(cv_image)
    if bbox_info:
        id, x, y, w, h, r = bbox_info
        points = get_rotated_box_points(x, y, w, h, -r)
        cv2.polylines(cv_image, [points], isClosed=True, color=(255, 0, 255), thickness=3)
        cv2.putText(cv_image, str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Cropping with buffer
        buffer = 20  # Adjust buffer size as needed
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        x_min = max(x_min - buffer, 0)
        y_min = max(y_min - buffer, 0)
        x_max = min(x_max + buffer, cv_image.shape[1])
        y_max = min(y_max + buffer, cv_image.shape[0])
        cropped_image = cv_image[y_min:y_max, x_min:x_max]

        try:
            crop_pub.publish(bridge.cv2_to_imgmsg(cropped_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

    try:
        image_pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
        print(e)

if __name__ == '__main__':
    rospy.init_node('image_processor', anonymous=True)
    image_sub = rospy.Subscriber("/oak/right/image_raw", Image, image_callback)
    image_pub = rospy.Publisher("/oak/right/annotated", Image, queue_size=10)
    crop_pub = rospy.Publisher("/oak/right/cropped", Image, queue_size=10)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")