#THIS FOR TESTING ON VEHICLE - HAS VIDEO AND IMAGE TESTING


from ultralytics import YOLO
from PIL import Image
import cv2
import sys
MODEL_WEIGHT_PATH = 'last.pt'
model = YOLO(MODEL_WEIGHT_PATH)

def empty_detect(img : cv2.Mat, empty_spot = 0):
    global model
    results = model(img)
    bboxes = []

    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        if class_id == empty_spot:
            bboxes.append(box.xywh[0].tolist())
    # results = model.predict(source=img)
    # return results[0].boxes.xywh
    return bboxes


def main(fn):
    image = cv2.imread(fn)
    bboxes = empty_detect(image)
    print("Detected",len(bboxes),"empty spot")
    for bb in bboxes:
        x,y,w,h = bb
        if not isinstance(x,(int,float)) or not isinstance(y,(int,float)) or not isinstance(w,(int,float)) or not isinstance(h,(int,float)):
            print("WARNING: make sure to return Python numbers rather than PyTorch Tensors")
        print("Corner",(x,y),"size",(w,h))
        cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 255), 3)
    cv2.imshow('Results', image)
    cv2.waitKey(0)


def main_webcam(): 
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    print("Press space to exit")
    while True:
        _, image = cap.read()
        
        bboxes = empty_detect(image)
        for bb in bboxes:
            x,y,w,h = bb
            cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 255), 3)
              
        cv2.imshow('Empty Spot Detection', image)     
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()



if __name__ == '__main__':
    fn = sys.argv[1]
    if fn != 'webcam':
        main(fn)
    else:
        main_webcam()