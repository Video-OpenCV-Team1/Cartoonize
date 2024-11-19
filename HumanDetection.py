import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np

def detect_people_and_generate_matrix(image, threshold=0.5):
    # 이미지를 텐서로 변환
    image_tensor = F.to_tensor(image).unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    # 모델 추론
    with torch.no_grad():
        predictions = model(image_tensor)

    # 이미지 크기의 빈 Matrix 생성
    h, w = image.shape[:2]
    return_matrix = np.zeros((h, w), dtype=np.int32)

    # 각 탐지된 객체에 대해 사람이면 고유 ID 부여
    person_id = 1
    for i, label in enumerate(predictions[0]['labels']):
        if label == 1 and predictions[0]['scores'][i] > threshold:  # 사람인지 확인
            mask = predictions[0]['masks'][i, 0].cpu().numpy()  # 마스크 추출
            return_matrix[mask > 0.4] = person_id  # ID 부여
            person_id += 1  # 다음 사람 ID 증가

    return return_matrix

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # Mask R-CNN 모델 로드 (한 번만 로드하여 재사용)
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # 추론 모드로 전환

    # 테스트용 이미지 로드
    image_path = "test.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 함수 호출
    result = detect_people_and_generate_matrix(image)
    # 결과 시각화
    plt.imshow(result, cmap="jet")
    plt.colorbar()
    plt.title("Return Matrix (People IDs)")
    plt.show()
