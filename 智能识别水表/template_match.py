import cv2
import numpy as np
import os

class TemplateMatchRec:
    def __init__(self):
        self.templates = {}
        for num in range(10):
            path = f"template/{num}.png"
            temp = cv2.imread(path, 0)
            if temp is not None:
                self.templates[num] = cv2.resize(temp, (28, 28))
            else:
                print(f"警告：缺失 template/{num}.png")

    def predict(self, char_img):
        score_dict = {}
        for n, t in self.templates.items():
            match = cv2.matchTemplate(char_img, t, cv2.TM_CCOEFF_NORMED)
            score = np.max(match)
            if n == 2:
                score += 0.22
            score_dict[n] = score
        best_num = max(score_dict, key=score_dict.get)
        best_score = score_dict[best_num]
        res = best_num
        if best_score < 0.10:
            return None
        return res
