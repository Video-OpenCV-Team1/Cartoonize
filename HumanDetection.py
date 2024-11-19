import cv2
import numpy as np
import torch

def HumanDetect(image, model_path='yolov5s.pt'):
    
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = 0.5


    height, width, _ = image.shape

    result = model(image)
    detections = result.xyxy[0].cpu().numpy()

    person_matrix = np.zeros((height, width), dtype=int)

    person_id = 1

    for *box, conf, cls in detections:
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)

            person_matrix[y1:y2, x1:x2] = person_id
            person_id += 1

    return person_matrix
