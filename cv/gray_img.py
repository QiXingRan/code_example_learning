import cv2
from PIL import Image
import numpy as np

image = cv2.imread('../images/bicycle.JPG', cv2.IMREAD_GRAYSCALE)

cv2.imwrite('../images/gray_bicycle.JPG', image)

image = cv2.imread('../images/bicycle.JPG')
print(image.shape)
imageGray = cv2.imread('../images/gray_bicycle.JPG')
print(imageGray.shape)
imageGray2 = cv2.imread('../images/bicycle.JPG', cv2.IMREAD_GRAYSCALE)
print(imageGray2.shape)


img = Image.open('../images/bicycle.JPG')
img_array = np.array(img)
# 检查图片是否为彩色图像
if img_array.ndim == 3 and img_array.shape[2] == 3:
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    # 计算灰度值
    gray_array = 0.299 * r + 0.587 * g + 0.114 * b
    gray_array = gray_array.astype(np.uint8)
    gray_img = Image.fromarray(gray_array)
    gray_img.save('../images/gray2_bicycle.JPG')

