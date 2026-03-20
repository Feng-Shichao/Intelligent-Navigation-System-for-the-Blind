import cv2
import numpy as np

# 创建一个640x480的白色图像
image = np.ones((480, 640, 3), dtype=np.uint8) * 255

# 在图像上添加文本
text = "Test Image for Smart Blind Cane"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 0, 0)  # 黑色
thickness = 2

# 获取文本的大小
text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

# 计算文本的居中位置
text_x = (image.shape[1] - text_size[0]) // 2
text_y = (image.shape[0] + text_size[1]) // 2

# 在图像上绘制文本
cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

# 保存图像
cv2.imwrite('test_image.jpg', image)
print("测试图像已保存为 test_image.jpg")