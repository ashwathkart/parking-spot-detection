from ultralytics import YOLO
import cv2
import numpy as np
import sys

MODEL_WEIGHT_PATH = 'best_obb1.pt'
model = YOLO(MODEL_WEIGHT_PATH)

def get_rotated_box_points(x, y, width, height, angle):
    rectangle = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                          [width / 2, height / 2], [-width / 2, height / 2]])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotated_rectangle = np.dot(rectangle, rotation_matrix) + np.array([x, y])
    return np.int0(rotated_rectangle)

def empty_detect(img: cv2.Mat, empty_spot=0):
    global model
    results = model(img)
    bboxes = []
    for box in results[0].obb:
        class_id = int(box.cls[0].item())
        if class_id == empty_spot:
            x, y, w, h, r = box.xywhr[0].tolist()
            bboxes.append((x, y, w, h, r))
    return bboxes

def main(fn):
    image = cv2.imread(fn)
    bboxes = empty_detect(image)
    print("Detected", len(bboxes), "empty spot(s)")
    for bb in bboxes:
        x, y, w, h, r = bb
        points = get_rotated_box_points(x, y, w, h, -r) 
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 255), thickness=3)
    cv2.imshow('Results', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        main(fn)
    else:
        print("Please provide a filename as an argument.")

#TESTING 
# python obb_detect.py images/img1.jpg 
