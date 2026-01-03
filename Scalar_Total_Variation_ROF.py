import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte

# 讀取彩色圖片
img = img_as_float(imread('input.jpg'))
noisy = np.clip(img + 0.1 * np.random.randn(*img.shape), 0, 1)

u = noisy.copy()
p = np.zeros((u.shape[0], u.shape[1], 2, 3)) # (y, x, 2, rgb)
tau, sigma, weight = 0.125, 0.125, 0.1

for _ in range(200):
    # 梯度 (針對每個顏色通道獨立算)
    grad_u = np.zeros_like(p)
    grad_u[:-1,:,0,:] = u[1:,:,:] - u[:-1,:,:]
    grad_u[:,:-1,1,:] = u[:,1:,:] - u[:,:-1,:]
    
    p_new = p + sigma * grad_u
    
    # ★ 關鍵：axis=2 (只計算空間梯度長度，RGB 分開)
    norm = np.sqrt(np.sum(p_new**2, axis=2, keepdims=True))
    p = p_new / np.maximum(1, norm / weight)
    
    # 散度
    div_p = np.zeros_like(u)
    div_p[1:-1,:,:] += p[1:-1,:,0,:] - p[:-2,:,0,:]
    div_p[:,1:-1,:] += p[:,1:-1,1,:] - p[:,:-2,1,:]
    
    u = (u + tau * div_p + tau * noisy) / (1 + tau)

imsave('output_scalar.png', img_as_ubyte(np.clip(u, 0, 1)))
