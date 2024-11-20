import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QComboBox, QSlider, QHBoxLayout
)
from PyQt5.QtCore import Qt
from torchvision.models.detection import maskrcnn_resnet50_fpn
import HumanDetection
import Quantization
import pencilDraw

class CartoonizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cartoonizer App")
        self.resize(600, 400)

        self.type = 1
        self.imageLink = "rose.jpg"
        self.videoLink = "input.mp4"
        self.clusters = 10
        self.quant_level = 7

        # GUI Layout
        layout = QVBoxLayout()

        # 작업 유형 선택
        self.type_label = QLabel("Select Task:")
        layout.addWidget(self.type_label)
        self.type_selector = QComboBox()
        self.type_selector.addItems(["1: Real-time (Webcam)", "2: Process Image", "3: Process Video"])
        self.type_selector.currentIndexChanged.connect(self.change_task_type)
        layout.addWidget(self.type_selector)

        # 파일 선택 버튼
        self.file_button = QPushButton("Select File (Image/Video)")
        self.file_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_button)

        # 클러스터와 양자화 단계 조정 슬라이더
        self.slider_layout = QHBoxLayout()

        self.cluster_label = QLabel("Clusters:")
        self.cluster_slider = QSlider(Qt.Horizontal)
        self.cluster_slider.setRange(1, 20)
        self.cluster_slider.setValue(self.clusters)
        self.cluster_slider.valueChanged.connect(self.update_clusters)
        self.slider_layout.addWidget(self.cluster_label)
        self.slider_layout.addWidget(self.cluster_slider)

        self.quant_label = QLabel("Quant Level:")
        self.quant_slider = QSlider(Qt.Horizontal)
        self.quant_slider.setRange(1, 10)
        self.quant_slider.setValue(self.quant_level)
        self.quant_slider.valueChanged.connect(self.update_quant_level)
        self.slider_layout.addWidget(self.quant_label)
        self.slider_layout.addWidget(self.quant_slider)

        layout.addLayout(self.slider_layout)

        # Start Processing 버튼
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)

        # 로그 및 상태 표시
        self.log_label = QLabel("Status: Ready")
        layout.addWidget(self.log_label)

        self.setLayout(layout)

        # Initialize HumanDetection model
        HumanDetection.model = maskrcnn_resnet50_fpn(pretrained=True)
        HumanDetection.model.eval()

    def change_task_type(self, index):
        self.type = index + 1
        self.log_label.setText(f"Task type set to: {self.type}")

    def select_file(self):
        if self.type == 2:
            file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg)")
            if file_name:
                self.imageLink = file_name
                self.log_label.setText(f"Selected Image: {file_name}")
        elif self.type == 3:
            file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
            if file_name:
                self.videoLink = file_name
                self.log_label.setText(f"Selected Video: {file_name}")

    def update_clusters(self, value):
        self.clusters = value
        self.cluster_label.setText(f"Clusters: {value}")

    def update_quant_level(self, value):
        self.quant_level = value
        self.quant_label.setText(f"Quant Level: {value}")

    def start_processing(self):
        self.log_label.setText("Processing started...")
        if self.type == 1:
            self.process_webcam()
        elif self.type == 2:
            self.process_image()
        elif self.type == 3:
            self.process_video()
        self.log_label.setText("Processing completed!")

    def process_webcam(self):
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            while True:
                ret, frame = cam.read()
                if ret:

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

                    processed_frame = self.composite(frame)
                    cv2.imshow("Processed Frame", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        cam.release()
        cv2.destroyAllWindows()

    def process_image(self):
        try:
            image = cv2.imread(self.imageLink)
            processed_image = self.composite(image)
            cv2.imshow("Processed Image", processed_image)
            cv2.imwrite("Processed_Image.png", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            self.log_label.setText(f"Error: {e}")

    def process_video(self):
        try:
            video = cv2.VideoCapture(self.videoLink)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            output = cv2.VideoWriter('Processed_Video.mp4', fourcc, fps, (width, height))

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                processed_frame = self.composite(frame)
                output.write(processed_frame)

            video.release()
            output.release()
            self.log_label.setText("Video processing completed!")
        except Exception as e:
            self.log_label.setText(f"Error: {e}")

    def composite(self, image):
        mask = HumanDetection.detect_people_and_generate_matrix(image)
        mask = mask.astype(np.uint8)
        pencil_sketch = pencilDraw.process_image_to_sketch(image, mask)
        quant = Quantization.color_quantization(image, mask, self.clusters, self.quant_level)

        normalized_pencil_sketch = pencil_sketch.astype(np.float32) / 255.0
        normalized_pencil_sketch = cv2.merge([normalized_pencil_sketch] * 3)
        multiplied = (normalized_pencil_sketch * quant).astype(np.uint8)

        final_image = np.where(mask[:, :, None] == 0, image, multiplied)
        return final_image

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CartoonizerApp()
    window.show()
    sys.exit(app.exec_())
