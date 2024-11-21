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
        self.iterations = 10  # iterations로 변수 이름 수정
        self.clusters = 7  # clusters로 변수 이름 수정

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

        # 클러스터와 반복 횟수 조정 슬라이더
        self.slider_layout = QHBoxLayout()

        self.iteration_label = QLabel("Iterations:")
        self.iteration_slider = QSlider(Qt.Horizontal)
        self.iteration_slider.setRange(1, 20)
        self.iteration_slider.setValue(self.iterations)
        self.iteration_slider.valueChanged.connect(self.update_iterations)
        self.slider_layout.addWidget(self.iteration_label)
        self.slider_layout.addWidget(self.iteration_slider)

        self.cluster_label = QLabel("Clusters:")
        self.cluster_slider = QSlider(Qt.Horizontal)
        self.cluster_slider.setRange(1, 10)
        self.cluster_slider.setValue(self.clusters)
        self.cluster_slider.valueChanged.connect(self.update_clusters)
        self.slider_layout.addWidget(self.cluster_label)
        self.slider_layout.addWidget(self.cluster_slider)

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

    def update_iterations(self, value):
        self.iterations = value
        self.iteration_label.setText(f"Iterations: {value}")

    def update_clusters(self, value):
        self.clusters = value
        self.cluster_label.setText(f"Clusters: {value}")

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
        quant = Quantization.color_quantization(image, mask, self.iterations, self.clusters)

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
