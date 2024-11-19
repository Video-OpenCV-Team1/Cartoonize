"""
출처:
    https://medium.com/@Kavya2099/image-to-pencil-sketch-using-opencv-ec3568443c5e
    https://github.com/pythonlessons/background_removal/blob/main/pencilSketch.py
"""
import numpy as np

class PencilDraw:
    def __init__(self):
        self.image                  = None
        self.mask                   = None  # 0: 배경, 1: 사람
        self.gray_image             = None
        self.inverted_image         = None
        self.blurred_image          = None
        self.inverted_blurred_image = None
        self.sketch_image           = None

    def set_image(self, image: np.ndarray):
        """
        처리할 입력 이미지/영상을 설정함
        매개변수 이미지/영상: 입력 이미지/영상을 numpy 배열로 설정
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("입력은 numpy 배열이어야 합니다.")
        self.image = image

    def set_mask(self, mask: np.ndarray):
        """
        사람 영역 나타나는 마스크 설정
        매개변수 마스크: 0(배경), 1(사람)인 이진 마스크
        """
        if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8:
            raise ValueError("마스크는 uint8 형의 numpy 배열이어야 합니다.")
        if self.image is None:
            raise ValueError("마스크를 정의하기 전 이미지/영상가 있어야 합니다.")
        if mask.shape[:2] != self.image.shape[:2]:
            raise ValueError("마스크 차원과 이미지 차원이 일치해야 합니다.")
        self.mask = mask

    def convert_to_grayscale(self):
        """
        수동 공식을 사용하여 이미지/영상을 그레이스케일로 변환함
        """
        if self.image is None:
            raise ValueError("설정된 이미지/영상이 없습니다. set_image()를 사용하여 이미지/영상을 입력하세요.")
        self.gray_image = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    def create_inverted_image(self):
        """
        반전된 그레이스케일 이미지/영상을 생성함
        """
        if self.gray_image is None:
            raise ValueError("그레이스케일 이미지/영상을 사용할 수 없습니다. 먼저 convert_to_grayscale()을 호출하세요.")
        self.inverted_image = 255 - self.gray_image

    def apply_gaussian_blur(self, kernel_size=11, sigma=5):
        """
        반전된 그레이스케일 이미지/영상에 가우시안 블러를 적용함
        매개변수 kernel_size: 가우시안 커널 크기(홀수여야 함)
        매개변수 sigma: 가우시안의 표준 편차
        """
        if self.inverted_image is None:
            raise ValueError("반전된 이미지/영상을 사용할 수 없습니다. 먼저 create_inverted_image()를 호출하세요.")

        # 가우시안 커널 생성 및 블러 적용
        kernel = gaussian_kernel(kernel_size, sigma)
        self.blurred_image = apply_gaussian_filter(self.inverted_image, kernel)

    def invert_blurred_image(self):
        """
        블러 이미지/영상을 반전함
        """
        if self.blurred_image is None:
            raise ValueError("블러된 이미지/영상을 사용할 수 없습니다. 먼저 apply_gaussian_blur()를 호출하세요.")
        self.inverted_blurred_image = 255 - self.blurred_image

    def create_sketch(self):
        """
        스케치 이미지/영상을 생성함
        """
        if self.gray_image is None:
            raise ValueError("그레이스케일 이미지/영상을 먼저 생성하세요. convert_to_grayscale() 호출 필요.")
        if self.inverted_blurred_image is None:
            raise ValueError("반전된 블러 이미지/영상을 먼저 생성하세요. invert_blurred_image() 호출 필요.")
        if self.mask is None:
            raise ValueError("마스크가 설정되지 않았습니다. set_mask() 호출 필요.")

        # 스케치 계산
        with np.errstate(divide='ignore', invalid='ignore'):
            sketch = np.divide(self.gray_image, self.inverted_blurred_image + 1e-5)  # 0 나누기 방지
            sketch = (sketch * 255).clip(0, 255).astype(np.uint8)

        # 마스크된 영역에만 스케치를 적용
        self.sketch_image = np.where(self.mask == 1, sketch, self.gray_image)

    def get_sketch(self):
        """
        스케치 이미지/영상을 반환함
        리턴: 최종 스케치 이미지/영상을 반환하는 numpy 배열
        """
        if self.sketch_image is None:
            raise ValueError("스케치 이미지/영상을 사용할 수 없습니다. 먼저 create_sketch()를 호출하세요.")
        return self.sketch_image


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    2D 가우시안 커널을 생성함
    매개변수 size: 커널 크기(홀수여야 함)
    매개변수 sigma: 가우시안의 표준 편차
    리턴: 2D 가우시안 커널
    """
    if size % 2 == 0:
        raise ValueError("커널 크기는 홀수이어야 합니다.")
    k = size // 2
    x, y = np.mgrid[-k:k + 1, -k:k + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()  # 정규화
    return kernel


def apply_gaussian_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    입력 이미지/영상에 가우시안 필터를 적용함
    매개변수 image: 입력 이미지/영상(그레이스케일)
    매개변수 kernel: 가우시안 커널
    리턴: 필터링된 이미지/영상
    """
    kernel_size = kernel.shape[0]
    pad_width = kernel_size // 2
    padded_image = np.pad(image, pad_width, mode='reflect')  # 경계 문제 해결
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.sum(roi * kernel)
    return filtered_image


def process_image_to_sketch(image: np.ndarray, mask: np.ndarray, kernel_size: int = 11, sigma: float = 5) -> np.ndarray:
    """
    주어진 이미지를 스케치 이미지로 변환하는 전체 과정을 실행하는 함수
    매개변수:
        image: numpy.ndarray
            입력 이미지 (3채널 RGB 이미지)
        mask: numpy.ndarray
            사람 영역을 나타내는 이진 마스크 (0: 배경, 1: 사람)
        kernel_size: int
            가우시안 커널 크기 (기본값: 11, 홀수여야 함)
        sigma: float
            가우시안의 표준 편차 (기본값: 5)
    리턴:
        sketch_image: numpy.ndarray
            최종 스케치 이미지
    """
    pencil_draw = PencilDraw()

    # 이미지와 마스크 설정
    pencil_draw.set_image(image)
    pencil_draw.set_mask(mask)

    # 처리 단계 수행
    pencil_draw.convert_to_grayscale()
    pencil_draw.create_inverted_image()
    pencil_draw.apply_gaussian_blur(kernel_size=kernel_size, sigma=sigma)
    pencil_draw.invert_blurred_image()
    pencil_draw.create_sketch()

    # 결과 반환
    return pencil_draw.get_sketch()
