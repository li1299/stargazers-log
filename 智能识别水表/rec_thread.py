from PyQt5.QtCore import QThread, pyqtSignal
from rec_engine import AllRecEngine

class RecThread(QThread):
    signal_result = pyqtSignal(str, float)
    def __init__(self, img_path, algo_idx):
        super().__init__()
        self.img_path = img_path
        self.algo_idx = algo_idx
        self.engine = AllRecEngine()

    def run(self):
        res, cost = self.engine.recognize(self.img_path, self.algo_idx)
        self.signal_result.emit(res, cost)
