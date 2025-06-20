import os
import cv2
import torch
from ultralytics import YOLO

def load_model(weights_path: str, device: str = "cpu"):
    """
    加载本地 YOLOv8 权重文件，并将模型移动到指定设备（CPU 或 GPU）。
    传入绝对路径以避免 Ultralytics 再去联网下载。
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"找不到权重文件：{weights_path}\n请检查路径是否正确。")
    model = YOLO(weights_path)
    model.to(device)
    return model

def clamp_and_shrink_box(x1, y1, x2, y2, width, height, margin=5):
    """
    1) 先将框坐标裁剪到 [0,width-1] 和 [0,height-1] 之间，避免超边界。
    2) 然后一并向内收缩 margin 像素，使得框本身略微小于原始预测框。
    """
    # 裁剪到边界内
    x1_clamped = max(0, min(int(x1), width - 1))
    y1_clamped = max(0, min(int(y1), height - 1))
    x2_clamped = max(0, min(int(x2), width - 1))
    y2_clamped = max(0, min(int(y2), height - 1))

    # 向内收缩 margin 像素
    x1_shrunk = x1_clamped + margin
    y1_shrunk = y1_clamped + margin
    x2_shrunk = x2_clamped - margin
    y2_shrunk = y2_clamped - margin

    # 再次确保不出现负宽高
    if x2_shrunk <= x1_shrunk:
        x2_shrunk = x1_shrunk + 1
    if y2_shrunk <= y1_shrunk:
        y2_shrunk = y1_shrunk + 1

    return x1_shrunk, y1_shrunk, x2_shrunk, y2_shrunk

def draw_with_margin(frame, results, color=(0, 255, 0), line_thickness=1, font_scale=0.4, margin=5):
    """
    将检测到的框画到原始帧上，并在框内部预留 margin 空间以显示标签文字。
    """
    height, width = frame.shape[:2]
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()  # 每个框的 raw (x1,y1,x2,y2)，相对于原始 frame
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
        cls = int(cls)
        label = model.names[cls]  # 例如 'stage1','stage2',...,'unstageable'
        caption = f"{label} {conf:.2f}"

        # 1) 裁剪并向内收缩 margin
        xx1, yy1, xx2, yy2 = clamp_and_shrink_box(x1, y1, x2, y2, width, height, margin=margin)

        # 2) 画矩形框
        cv2.rectangle(
            frame,
            (xx1, yy1),
            (xx2, yy2),
            color,
            line_thickness
        )

        # 3) 确保标签文字位于框内部，如果 y1 太靠上，就向下偏移；如果框太小则强制置于内部
        ((text_w, text_h), _) = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = xx1
        text_y = yy1 - 3  # 默认在框上方留 3 个像素

        # 如果框太贴顶，label 会越界，改画在框内部稍往下
        if text_y - text_h < 0:
            text_y = yy1 + text_h + 3
        # 如果文字宽度超出右边，则左移文字
        if text_x + text_w >= width:
            text_x = width - text_w - 1

        # 4) 在框内或框上方画标签背景（可选，提升可读性）
        cv2.rectangle(
            frame,
            (text_x - 1, text_y - text_h - 1),
            (text_x + text_w + 1, text_y + 1),
            color,
            thickness=-1  # 填充背景色
        )
        # 5) 写文字（白色），以便对比度更高
        cv2.putText(
            frame,
            caption,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # 白色文字
            1,
            cv2.LINE_AA
        )

    return frame

def realtime_camera_detection(model, device: str = "cpu",
                              source: int = 0,
                              imgsz: int = 640,
                              conf_threshold: float = 0.25,
                              use_plot: bool = False,
                              display_scale: float = 0.7,
                              margin: int = 5):
    """
    使用 OpenCV 打开摄像头，对每一帧进行 YOLOv8 推理，将检测结果画到画面中并实时显示。

    参数：
      - model: 已加载的 YOLOv8 模型 (YOLO 类实例)。
      - device: "cpu" 或 "cuda"，指定模型推理设备。
      - source: 摄像头索引 (一般 0 表示默认摄像头)，也可传入视频文件路径，但此处用于实时摄像头。
      - imgsz: 将每帧缩放至 imgsz×imgsz 后再映射回原始尺寸。
      - conf_threshold: 置信度阈值，小于此值的预测会被过滤掉。
      - use_plot: True 时直接使用 results[0].plot()；False 时使用自定义 draw_with_margin()。
      - display_scale: 显示时将完整帧统一缩放的比例(<1 可完整显示全部)。
      - margin: 缩小框的像素数目，让标签在边界可见。
    """
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # Windows 下推荐加 CAP_DSHOW
    if not cap.isOpened():
        print(f"[Error] 无法打开摄像头: {source}")
        return

    # 获取摄像头的帧率，以便控制显示速度
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps and fps > 0 else 1

    window_name = "Real-Time PU Detection (q to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print(f">>> 摄像头已打开 (索引={source})，按 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(">>> 读取摄像头帧失败，尝试重连...")
            cap.release()
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            continue

        # YOLOv8 推理 (返回的 xyxy 坐标是针对原始 frame 分辨率)
        results = model(frame, imgsz=imgsz, device=device, conf=conf_threshold)

        if use_plot:
            # 方式一：直接使用 Ultralytics 自带 plot()
            annotated_frame = results[0].plot()
        else:
            # 方式二：裁剪 + 缩小 + 绘制带背景的标签
            annotated_frame = frame.copy()
            annotated_frame = draw_with_margin(
                annotated_frame,
                results,
                color=(0, 255, 0),
                line_thickness=1,
                font_scale=0.4,
                margin=margin
            )

        # 缩放显示，保证可以完整看到整个画面和框
        if display_scale != 1.0:
            h, w = annotated_frame.shape[:2]
            new_w, new_h = int(w * display_scale), int(h * display_scale)
            display_frame = cv2.resize(annotated_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            display_frame = annotated_frame

        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            print(">>> 收到退出指令，停止实时检测。")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 1. 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> 使用设备：{device}")

    # 2. 微调后的权重绝对路径，请改为你本地路径
    WEIGHTS_PATH = "E:/Image_PI_identify/runs/detect/train3/weights/best.pt"

    # 加载模型
    model = load_model(WEIGHTS_PATH, device)

    # 3. 摄像头索引，通常 0 表示默认摄像头；如有多路摄像头可改为 1、2...
    CAMERA_INDEX = 0

    # 4. 是否使用 Ultralytics 自带 plot()？False 则使用自定义裁剪 + 缩小 + 绘制
    USE_PLOT = False

    # 5. 缩放显示比例 (<1 可完整显示全部)，设置为 1.0 则按原始大小显示
    DISPLAY_SCALE = 0.7

    # 6. 给每个检测框向内缩 margin 像素，让标签在边界可见
    MARGIN = 5

    # 启动摄像头实时检测
    realtime_camera_detection(
        model=model,
        device=device,
        source=CAMERA_INDEX,
        imgsz=640,
        conf_threshold=0.25,
        use_plot=USE_PLOT,
        display_scale=DISPLAY_SCALE,
        margin=MARGIN
    )
