from preprocess import WaterMeterPreprocess
from template_match import TemplateMatchRec
from hog_svm import SVMRec
from cnn_rec import CNNRec
import cv2

class AllRecEngine:
    def __init__(self):
        self.pre = WaterMeterPreprocess()
        self.temp = TemplateMatchRec()
        self.svm = SVMRec()
        self.cnn = CNNRec()

    def recognize(self, img_path, algo_idx):
        import time
        t0 = time.time()
        img = cv2.imread(img_path)
        if img is None:
            return "图片读取失败", round(time.time()-t0,2)
        img = self.pre.adjust_light(img)
        bin_img = self.pre.binary_segment(img)
        dial = self.pre.locate_dial(bin_img, img)
        char_with_x = self.pre.split_char(dial)
        if not char_with_x:
            return "未识别到有效数字", round(time.time()-t0,2)
        right_char, x_pos = char_with_x[-1]
        num = None
        if algo_idx == 0:
            num = self.temp.predict(right_char)
        elif algo_idx == 1:
            num = self.svm.predict(right_char)
        else:
            num = self.cnn.predict(right_char)
        if num is not None:
            res = str(num)
        else:
            res = "未识别到有效数字"
        cost = round(time.time() - t0, 2)
        del img, bin_img, dial
        return res, cost
