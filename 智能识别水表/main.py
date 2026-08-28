import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from rec_thread import RecThread

os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=freetype"

class WaterMeterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水表读数识别系统｜分文件工程版")
        self.resize(1100, 700)
        self.img_path = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(10,10,10,10)

        self.btn_upload = QPushButton("上传水表图片")
        self.btn_upload.setFixedHeight(40)
        self.btn_upload.clicked.connect(self.upload_img)
        left_layout.addWidget(self.btn_upload)

        left_layout.addWidget(QLabel("选择识别算法："))
        self.cbx_algo = QComboBox()
        self.cbx_algo.addItems([
            "算法1：模板匹配",
            "算法2：HOG+SVM",
            "算法3：轻量化CNN"
        ])
        left_layout.addWidget(self.cbx_algo)

        self.btn_run = QPushButton("开始识别")
        self.btn_run.setFixedHeight(40)
        self.btn_run.clicked.connect(self.start_rec)
        left_layout.addWidget(self.btn_run)

        left_layout.addWidget(QLabel("识别读数（仅最后一位整数）："))
        self.label_res = QLabel("读数：")
        self.label_res.setStyleSheet("font-size:24px;color:#0044cc;font-weight:bold;")
        left_layout.addWidget(self.label_res)

        left_layout.addWidget(QLabel("处理耗时(s)："))
        self.label_time = QLabel("0.00")
        self.label_time.setStyleSheet("font-size:18px;")
        left_layout.addWidget(self.label_time)

        left_layout.addStretch()
        main_layout.addLayout(left_layout, 1)

        right_layout = QVBoxLayout()
        self.label_img = QLabel("图片预览区域")
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setMinimumSize(700,600)
        self.label_img.setStyleSheet("border:1px solid #cccccc;background:#f5f5f5;")
        right_layout.addWidget(self.label_img)
        main_layout.addLayout(right_layout, 3)

    def upload_img(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择水表图片", "", "图片(*.jpg *.png *.jpeg *.bmp)")
        if path:
            self.img_path = path
            pix = QPixmap(path).scaled(self.label_img.size(), Qt.KeepAspectRatio)
            self.label_img.setPixmap(pix)
            self.label_res.setText("读数：")
            self.label_time.setText("0.00")

    def start_rec(self):
        if not self.img_path:
            QMessageBox.warning(self, "提示", "请先上传水表图片！")
            return
        algo_idx = self.cbx_algo.currentIndex()
        self.label_res.setText("识别中...")
        self.rec_thread = RecThread(self.img_path, algo_idx)
        self.rec_thread.signal_result.connect(self.show_result)
        self.rec_thread.start()

    def show_result(self, res, cost):
        self.label_res.setText(f"读数：{res}")
        self.label_time.setText(f"{cost}")
        if float(cost) > 3:
            QMessageBox.information(self, "速度提示", f"处理耗时{cost}s，超过3秒限制")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WaterMeterGUI()
    win.show()
    sys.exit(app.exec_())
