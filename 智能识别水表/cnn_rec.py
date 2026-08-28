import numpy as np
import os

class CNNRec:
    def __init__(self):
        self.weight_path = "cnn_weights.npy"
        self.weights_exist = os.path.exists(self.weight_path)
        if self.weights_exist:
            w_data = np.load(self.weight_path, allow_pickle=True).item()
            self.conv1 = w_data["conv1"]
            self.bias1 = w_data["bias1"]
            self.fc1 = w_data["fc1"]
            self.bias_fc1 = w_data["bias_fc1"]
            self.fc2 = w_data["fc2"]
            self.bias_fc2 = w_data["bias_fc2"]
            print("CNN权重加载完成")
        else:
            print("CNN权重缺失，CNN不可用")

    def conv2d(self, x, kernel, bias):
        batch, h_img, w_img, c = x.shape
        k_h, k_w, _, out_c = kernel.shape
        out_h = h_img - k_h + 1
        out_w = w_img - k_w + 1
        out = np.zeros((batch, out_h, out_w, out_c))
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:,i:i+k_h,j:j+k_w,:]
                for k_idx in range(out_c):
                    out[:,i,j,k_idx] = np.sum(patch * kernel[:,:,:,k_idx], axis=(1,2,3)) + bias[k_idx]
        return out

    def max_pool(self, x):
        return np.max(x.reshape(x.shape[0],x.shape[1]//2,2,x.shape[2]//2,2,x.shape[3]), axis=(2,4))

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, char_img):
        if not self.weights_exist:
            return 0
        img_norm = char_img / 255.0
        img_input = img_norm.reshape(1,28,28,1)
        conv_out = self.conv2d(img_input, self.conv1, self.bias1)
        pool_out = self.max_pool(conv_out)
        flat = pool_out.reshape(1, -1)
        fc1_out = np.dot(flat, self.fc1) + self.bias_fc1
        fc1_out = self.relu(fc1_out)
        pred = np.dot(fc1_out, self.fc2) + self.bias_fc2
        out = int(np.argmax(pred, axis=1)[0])
        return out
