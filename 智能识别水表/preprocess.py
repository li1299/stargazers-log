import cv2
import numpy as np

class WaterMeterPreprocess:
    def adjust_light(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    def binary_segment(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 4)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_img = cv2.dilate(bin_img, kernel_dilate, iterations=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        return bin_img

    def locate_dial(self, bin_img, origin_img):
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wheel_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h
            if 50 < w < 400 and 15 < h < 90 and 0.35 < ratio < 0.95:
                wheel_candidates.append((x, y, w, h))
        if not wheel_candidates:
            return origin_img
        wheel_candidates.sort(key=lambda r: (r[0] + r[2], r[2] * r[3]), reverse=True)
        x, y, w, h = wheel_candidates[0]
        crop_x1 = max(0, x - 12)
        crop_y1 = max(0, y - 12)
        crop_x2 = min(origin_img.shape[1], x + w + 16)
        crop_y2 = min(origin_img.shape[0], y + h + 16)
        dial_region = origin_img[crop_y1:crop_y2, crop_x1:crop_x2]
        return dial_region

    def split_char(self, dial_img):
        gray = cv2.cvtColor(dial_img, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            w_h_ratio = w / h
            if 8 < w < 50 and 18 < h < 65 and 0.28 < w_h_ratio < 0.85:
                char_boxes.append((x, y, w, h))
        char_boxes.sort(key=lambda b: b[0])
        char_list = []
        for box in char_boxes:
            x, y, w, h = box
            crop = gray[y:y + h, x:x + w]
            resize_char = cv2.resize(crop, (28, 28))
            char_list.append((resize_char, x))
        return char_list
