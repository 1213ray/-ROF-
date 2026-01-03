import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_ubyte

# 讀取並轉灰階
img = img_as_float(rgb2gray(imread('input.jpg')))
noisy = img + 0.1 * np.random.randn(*img.shape) # 加雜訊

# 參數與變數初始化
u = noisy.copy()
p = np.zeros((u.shape[0], u.shape[1], 2)) # 對偶變數 (y, x, 2)
tau, sigma, weight = 0.125, 0.125, 0.1

# Primal-Dual 迭代
for _ in range(100):
    # 計算梯度 (Gradient)
    grad_u = np.zeros_like(p)
    grad_u[:-1, :, 0] = u[1:, :] - u[:-1, :] # dy
    grad_u[:, :-1, 1] = u[:, 1:] - u[:, :-1] # dx
    
    # 更新 P (投影)
    p_new = p + sigma * grad_u
    norm = np.sqrt(np.sum(p_new**2, axis=2, keepdims=True))
    p = p_new / np.maximum(1, norm / weight)
    
    # 計算散度 (Divergence)
    div_p = np.zeros_like(u)
    div_p[1:-1, :] += p[1:-1, :, 0] - p[:-2, :, 0]
    div_p[:, 1:-1] += p[:, 1:-1, 1] - p[:, :-2, 1]
    
    # 更新 U
    u = (u + tau * div_p + tau * noisy) / (1 + tau)

imsave('output_gray.png', img_as_ubyte(np.clip(u, 0, 1)))
