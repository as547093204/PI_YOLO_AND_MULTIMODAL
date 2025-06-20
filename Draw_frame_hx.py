import os
import cv2

def extract_frames_from_video(video_path, output_dir, segments=3, frames_per_segment=1):
    """
    从给定视频中按“前、中、后”三个阶段各抽取若干帧并保存为图片。
    video_path: 视频文件的完整路径
    output_dir: 抽取到的图片保存目录
    segments:   要划分的视频阶段数，默认为3（前段/中段/后段）
    frames_per_segment: 每个阶段要抽取的帧数，默认为1
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频：{video_path}")
        return

    # 视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"[警告] 视频帧数异常：{video_path}")
        cap.release()
        return

    # 每个阶段的长度（向下取整）
    segment_length = total_frames // segments

    # 存储最终要抽取的帧索引
    frame_indices = []

    for i in range(segments):
        start = i * segment_length
        # 最后一个阶段，要覆盖到最后一帧
        if i == segments - 1:
            length = total_frames - start
        else:
            length = segment_length

        # 在当前阶段中，均匀选取 frames_per_segment 帧
        # 这里使用 (j+1)/(frames_per_segment+1) 的方式，避免直接取阶段首尾
        for j in range(frames_per_segment):
            idx = start + int((j + 1) * length / (frames_per_segment + 1))
            # 边界检查
            if idx < start:
                idx = start
            if idx >= start + length:
                idx = start + length - 1
            frame_indices.append(idx)

    # 去重并排序
    frame_indices = sorted(set(frame_indices))

    # 逐帧读取并保存
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[错误] 无法读取视频 {video_name} 的第 {idx} 帧")
            continue

        # 生成保存的文件名，例如：videoA_frame_150.jpg
        img_filename = f"{video_name}_frame_{idx}.jpg"
        save_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(save_path, frame)

    cap.release()
    print(f"[完成] 已从“{video_name}”抽取 {len(frame_indices)} 帧，保存到 {output_dir}")


def main():
    # ———— 配置区 ————
    # 1. 输入视频文件夹路径（请改为你的实际路径）
    input_dir = r"data\input\your_path"
    # 2. 输出图片保存路径（请改为你的实际路径）
    output_dir = r"data\output\your_path"

    # 创建输出目录（如果不存在就新建）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 支持的常见视频格式扩展名
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

    # 遍历输入文件夹下的所有文件
    for filename in os.listdir(input_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in video_exts:
            continue  # 不是视频文件则跳过

        video_path = os.path.join(input_dir, filename)
        extract_frames_from_video(video_path, output_dir)


if __name__ == "__main__":
    main()
