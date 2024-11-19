import cv2
import numpy as np
import HumanDetection

imageLink = 'input.png'

# True : 실시간 이미지 처리, False : input.png 이미지 처리
isRealTime = False

def cartoonize(image):
    print('Cartoonizing...')
    return image

def get_binary_image(human_detection):
    tmp = np.zeros((human_detection.size, human_detection[0].size, 3), np.uint8)
    for i, row in enumerate(human_detection):
        for j, pixel in enumerate(human_detection[i]):
            if human_detection[i][j] != 0:
                tmp[i][j] = [255, 255, 255]
    return tmp

if isRealTime:
    cam = cv2.VideoCapture(0)

    if cam.isOpened():
        cv2.namedWindow('Cartoon')
        cv2.namedWindow('Cam')
        while True:
            ret, frame = cam.read()
            if ret:
                if cv2.waitKey(1) != -1:
                    break

                if cv2.waitKey(1) == ord('s'):
                    break

                # 사람 인식 확인용
                # binary = HumanDetection.HumanDetect(frame)
                # tmp = get_binary_image(binary)
                #
                # cv2.imshow('Cartoon', tmp)
                cv2.imshow('Cam', frame)
            else:
                break

    cam.release()
else:
    try:
        image = cv2.imread(imageLink)
        #cv2.imshow('image', image)

    except:
        print(imageLink + '을/를 찾을 수 없습니다.')
        quit()

    # hExist = HumanDetection.HumanDetect(image)
    # cv2.imwrite('Output.png', hExist)


cv2.destroyAllWindows()