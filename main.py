import cv2
import HumanDetection

imageLink = 'input.png'
isRealTime = False

def cartoonize(image):
    print('Cartoonizing...')
    return image

if isRealTime:
    cam = cv2.VideoCapture(0)

    if cam.isOpened():
        cv2.namedWindow('Cartoon')
        while True:
            ret, frame = cam.read()
            if ret:
                if cv2.waitKey(1) != -1:
                    break

                if cv2.waitKey(1) == ord('s'):
                    break

                binary = HumanDetection.HumanDetect(frame)

                cv2.imshow('Cartoon', frame)
            else:
                break

    cam.release()
else:
    try:
        image = cv2.imread(imageLink)
        cv2.imshow('image', image)
    except:
        print(imageLink + '을/를 찾을 수 없습니다.')
        quit()

    cartoonize(image)

cv2.destroyAllWindows()