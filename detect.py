import torch
from ultralytics import YOLO
import cv2
from dataclasses import dataclass
import time
import logging
import pickle

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class Config:
    source: str = "new_park.png"
    view_img: bool = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    skip: int = 1
    yolo: bool = True
    yolo_type = "yolov8m.pt"


class Detector:
    def __init__(self, pos_list_file):
        self.config = Config()
        self.device = self.config.device
        self.start = time.time()
        self.count = 0
        with open(pos_list_file, "rb") as f:
            self.pos_list = pickle.load(f)

    def load_model(self):
        if self.config.yolo:
            if self.config.yolo_type is None or self.config.yolo_type == "":
                raise ValueError("YOLO model type is not specified")
            model = YOLO(self.config.yolo_type)
            logging.info(f"YOLOv8 Inference using {self.config.yolo_type}")
        return model

    def detect(self):
        model = self.load_model()
        img = cv2.imread(self.config.source)
        outputs = model(img)
        if self.config.view_img:
            if self.config.yolo:
                annotated_frame = outputs[0].plot()
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                cv2.waitKey(0)
        logging.info("************************* Done *****************************")

    def iou(self, boxA, boxB):
        boxB = (
            min(boxB[0][0], boxB[1][0]),
            min(boxB[0][1], boxB[1][1]),
            max(boxB[0][0], boxB[1][0]),
            max(boxB[0][1], boxB[1][1]),
        )
        # box format: (x_min, y_min, x_max, y_max)
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        union_area = boxA_area + boxB_area - intersection_area
        iou = intersection_area / union_area
        return iou

    def detect_and_check_iou(self):
        model = self.load_model()
        img = cv2.imread(self.config.source)
        outputs = model(img)
        # create a list of colors to use for each bounding box
        outputss = outputs[0].boxes
        colors = ["green"] * len(outputss)

        for i, output in enumerate(outputss):
            # get the detection box coordinates
            box = output.boxes.data.tolist()
            box = tuple(output.xyxy.tolist()[0])

            # iterate over the boxes from pos_list_file
            for j, pos_box in enumerate(self.pos_list):
                iou = self.iou(box, pos_box)

                p1, p2 = pos_box
                if iou > 0.4:
                    color = (0, 0, 255)
                    del self.pos_list[j]
                    text = "occupied"
                    cv2.putText(
                        img,
                        text,
                        (p1[0], p1[1] - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.5,
                        color,
                        1,
                    )
                else:
                    color = (0, 255, 0)

                cv2.rectangle(img, p1, p2, color, 2)

        cv2.imshow("image", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    pos_list_file = "assets/park_positions"
    detector = Detector(pos_list_file)
    detector.detect_and_check_iou()
