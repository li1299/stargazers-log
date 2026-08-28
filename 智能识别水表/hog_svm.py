from skimage.feature import hog
from sklearn.svm import SVC
import joblib

class SVMRec:
    def __init__(self):
        self.model = SVC(kernel="rbf")
        try:
            self.model = joblib.load("svm_model.pkl")
        except:
            print("警告：缺失 svm_model.pkl")

    def get_hog(self, img):
        return hog(img, orientations=9, pixels_per_cell=(3,3), cells_per_block=(2,2)).reshape(1,-1)

    def predict(self, char_img):
        feat = self.get_hog(char_img)
        out = int(self.model.predict(feat)[0])
        return out
