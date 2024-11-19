import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Mask R-CNN 모델 로드 (사전 학습된 COCO 데이터셋 사용)
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 추론 모드로 전환


def cartoonize_function(image):
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 엣지 검출
    edges = cv2.Canny(gray, 100, 200)

    # Bilateral Filter로 색상 영역 보존
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # 엣지와 결합
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(filtered, edges_colored)
    return cartoon


# 2. 입력 이미지 로드 및 전처리
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 로드하므로 RGB로 변환
    return image


# 3. 사람만 추출
def get_person_masks(image, threshold=0.5):
    # 이미지 전처리
    image_tensor = F.to_tensor(image).unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    # 모델 추론
    with torch.no_grad():
        predictions = model(image_tensor)

    # 사람 클래스 필터링 (COCO dataset에서 사람 클래스는 label 1)
    masks = []
    for i, label in enumerate(predictions[0]['labels']):
        if label == 1 and predictions[0]['scores'][i] > threshold:
            masks.append(predictions[0]['masks'][i, 0].cpu().numpy())  # 1st dimension mask

    return masks


# 4. Return Matrix 생성
def generate_return_matrix(image, masks):
    return_matrix = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    for idx, mask in enumerate(masks, start=1):
        return_matrix[mask > 0.5] = idx  # Threshold로 binary mask 생성
    return return_matrix


# 5. Cartoonize 사람만 처리
def cartoonize_person(image, person_matrix):
    output_image = np.zeros_like(image)
    for person_id in range(1, np.max(person_matrix) + 1):
        person_mask = (person_matrix == person_id)
        person_area = image.copy()
        person_area[~person_mask] = 0  # 배경 제거
        cartoonized_person = cartoonize_function(person_area)  # Cartoonize 처리 함수 (직접 구현 필요)
        output_image[person_mask] = cartoonized_person[person_mask]
    return output_image


# 6. 테스트 실행
if __name__ == "__main__":
    # 이미지 로드
    image_path = "test.jpg"
    image = load_image(image_path)

    # 사람 마스크 추출
    masks = get_person_masks(image)

    # Return Matrix 생성
    return_matrix = generate_return_matrix(image, masks)

    # Return Matrix 시각화 (선택)
    plt.imshow(return_matrix, cmap='jet')
    plt.title("Return Matrix")
    plt.show()

    # Cartoonize 적용
    cartoonized_image = cartoonize_person(image, return_matrix)

    # 결과 저장 또는 시각화
    plt.imshow(cartoonized_image)
    plt.title("Cartoonized Image")
    plt.axis("off")
    plt.show()
