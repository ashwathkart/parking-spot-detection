import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import numpy as np

MODEL_WEIGHT_PATH = 'best_obb1.pt'
model = YOLO(MODEL_WEIGHT_PATH)
bridge = CvBridge()

def get_rotated_box_points(x, y, width, height, angle):
    rectangle = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                          [width / 2, height / 2], [-width / 2, height / 2]])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotated_rectangle = np.dot(rectangle, rotation_matrix) + np.array([x, y])
    return np.int0(rotated_rectangle)

def process_image(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    except CvBridgeError as e:
        print(e)
        return
    
    bboxes = empty_detect(cv_image)
    for bb in bboxes:
        x, y, w, h, r = bb
        points = get_rotated_box_points(x, y, w, h, -r)
        cv2.polylines(cv_image, [points], isClosed=True, color=(255, 0, 255), thickness=3)
    
    try:
        img_msg_out = bridge.cv2_to_imgmsg(cv_image, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    pub.publish(img_msg_out)

def empty_detect(img):
    global model
    results = model(img)
    bboxes = []
    for box in results[0].obb:
        class_id = int(box.cls[0].item())
        if class_id == 0:  # Assuming 0 is the class for empty spots
            x, y, w, h, r = box.xywhr[0].tolist()
            bboxes.append((x, y, w, h, r))
    return bboxes

if __name__ == '__main__':
    rospy.init_node('image_processor', anonymous=True)
    sub = rospy.Subscriber("/oak/right/image_raw", Image, process_image)
    pub = rospy.Publisher("/oak/right/annotated", Image, queue_size=10)
    rospy.spin()
