import matplotlib.pyplot as plt
from ultralytics import YOLO

# —— 设置中文显示，避免 Matplotlib 中文乱码 ——
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# —— 加载 YOLOv8m 模型结构（仅加载配置，不加载权重） ——
model = YOLO('E:/Image_PI_identify/yolov8m.pt')    # 如果你已经有 .pt 权重，也可以写成 'yolov8m.pt' 或者本地路径

# —— 绘制并获取 PIL.Image 对象 ——
architecture_img = model.plot()

# —— 用 Matplotlib 显示网络结构图 ——
plt.figure(figsize=(12, 12))
plt.imshow(architecture_img)
plt.axis('off')
plt.title('YOLOv8m 模型结构')
plt.show()
