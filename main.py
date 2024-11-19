import cv2
import numpy as np
import HumanDetection
import Quantization
import pencilDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn

imageLink = 'rose.jpg'
videoLink = 'input.mp4'

# 1 : 실시간 이미지 처리, 2 : input.png 이미지 처리, 3 : input.mp4 영상 처리
type = 3

def cartoonize(image):
    print('Cartoonizing...')
    return image

def get_binary_image(human_detection):
    # numpy 배열로 변환하여 계산을 벡터화
    tmp = np.zeros((human_detection.shape[0], human_detection.shape[1], 3), dtype=np.uint8)

    # human_detection 배열을 0이 아닌 값들에 대해서 처리
    mask = human_detection != 0
    tmp[mask] = np.array([human_detection[mask] / 8 * 255, human_detection[mask] / 8 * 255, np.full_like(human_detection[mask], 255)]).T

    return tmp

def composite(image):
    mask = HumanDetection.detect_people_and_generate_matrix(image)
    mask = mask.astype(np.uint8)
    pencil_sketch = pencilDraw.process_image_to_sketch(image, mask)
    quant = Quantization.color_quantization(image, mask, 10, 7)

    normalized_pencil_sketch = pencil_sketch.astype(np.float32) / 255.0
    normalized_pencil_sketch = cv2.merge([normalized_pencil_sketch] * 3)
    multiplied = (normalized_pencil_sketch * quant).astype(np.uint8)

    final_image = np.where(mask[:, :, None] == 0, image, multiplied)
    cv2.imshow('final_image', final_image)

    return final_image

# Mask R-CNN 모델 로드 (한 번만 로드하여 재사용)
HumanDetection.model = maskrcnn_resnet50_fpn(pretrained=True)
HumanDetection.model.eval()  # 추론 모드로 전환

if type == 1:
    cam = cv2.VideoCapture(0)

    if cam.isOpened():
        while True:
            ret, frame = cam.read()
            if ret:
                if cv2.waitKey(1) != -1:
                    break

                if cv2.waitKey(1) == ord('s'):
                    break

                # 사람 인식 확인용
                # binary = HumanDetection.detect_people_and_generate_matrix(frame)
                # clust = Quantization.color_quantization(frame, binary, 10, 7)
                # draw = pencilDraw.process_image_to_sketch(frame, binary)
                # tmp = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
                #
                # mask = draw > 240
                # tmp[mask] = np.array(
                #     [np.full_like(draw[mask], 1), np.full_like(draw[mask], 1), np.full_like(draw[mask], 1)]).T
                #
                # cv2.imshow('tmp', tmp)
                #
                # tmp = tmp * clust
                #
                # cv2.imshow('result', tmp)
                # cv2.imshow('Cartoon', clust)
                # cv2.imshow('Source', frame)
                draw = composite(frame)

                cv2.imshow('pencil', draw)
            else:
                break
    cam.release()
elif type == 2:
    try:
        image = cv2.imread(imageLink)
        #cv2.imshow('image', image)

    except:
        print(imageLink + '을/를 찾을 수 없습니다.')
        quit()

    clust = composite(image)

    cv2.imwrite('Output.png', clust)
    cv2.waitKey(0)
elif type == 3:
    try:
        video = cv2.VideoCapture(videoLink)
        if video.isOpened():

            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  #
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            length = video.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = video.get(cv2.CAP_PROP_FPS)  # 또는 cap.get(5)
            print('프레임 너비: %d, 프레임 높이: %d, 길이: %d, 초당 프레임 수: %d' % (width, height, length, fps))

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            output = cv2.VideoWriter('Output.mp4', fourcc, fps, (int(width), int(height)))

            while True:
                ret, frame = video.read()
                if ret :
                    r = composite(frame)
                    output.write(r)
                else:
                    print("비디오가 종료되었습니다.")
                    break
        else:
            print(videoLink + '을/를 찾을 수 없습니다.')
    except:
        print(videoLink + '을/를 찾을 수 없습니다.')
        quit()

    video.release()
    cv2.destroyAllWindows()

else:
    print('정확한 작업을 입력해주세요.')

cv2.destroyAllWindows()