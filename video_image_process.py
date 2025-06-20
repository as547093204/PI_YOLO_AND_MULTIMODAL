import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy as sp
import os
import skimage.color as color
from skimage.transform import pyramid_laplacian
import scipy.fftpack as fftpack
import scipy.signal as signal
from tqdm import tqdm

# 在文件开头添加以下代码
import time
import logging
from datetime import datetime
import psutil
import os


log_path = "E:/ulcer_classify/logs/Euler_amplify"
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_path}/cpu_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)


def log_performance(name, start_time):
    """记录性能指标"""
    duration = time.time() - start_time
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent()

    logging.info(f"\nPerformance metrics for {name}:")
    logging.info(f"Execution time: {duration:.2f} seconds")
    logging.info(f"Memory usage: {memory_mb:.2f} MB")
    logging.info(f"CPU usage: {cpu_percent:.2f}%")

    return duration, memory_mb, cpu_percent

#从视频文件读取指定范围的帧（默认从第一帧到最后一帧），并返回视频的帧率和帧数组（每一帧叠加的块（似高维张量））
# def getVideoFrames(videoFilePath, startFrameNumber=-1, endFrameNumber=-1):
#     frames = []
#     vidcap = cv2.VideoCapture(videoFilePath)
#     fps = vidcap.get(cv2.CAP_PROP_FPS)
#     totalFrame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
#     if startFrameNumber == -1:
#         startFrameNumber = 0
#     if endFrameNumber == -1:
#         endFrameNumber = totalFrame - 1
#     success, image = vidcap.read()
#     count = 0
#     success = True
#     while success:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = color.rgb2yiq(image).astype(np.float32)
#
#         if count < startFrameNumber:
#             success, image = vidcap.read()
#             count += 1
#             continue
#         elif count >= endFrameNumber:
#             break
#         else:
#             frames.append(image)
#         success, image = vidcap.read()
#         count += 1
#     frames = np.array(frames)
#
#     return fps, frames

def getVideoFrames(videoFilePath, startFrameNumber=-1, endFrameNumber=-1, target_size=(640, 480)):
    frames = []
    vidcap = cv2.VideoCapture(videoFilePath)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalFrame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if startFrameNumber == -1:
        startFrameNumber = 0
    if endFrameNumber == -1:
        endFrameNumber = totalFrame - 1

    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if not success:
            break
        if count < startFrameNumber:
            count += 1
            continue
        elif count > endFrameNumber:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)  # 统一修改为目标大小
        image = color.rgb2yiq(image).astype(np.float32)
        frames.append(image)
        count += 1

    frames = np.array(frames)
    return fps, frames

# 定义一个名为 compareImg 的函数，用于在同一个窗口中显示两张图片，
# 并在每张图片上标注出相同的矩形区域（Region of Interest，简称 ROI）。
def compareImg(leftImg, rightImg, roi, leftTitle, rightTitle):
    #接受五个参数：
    # leftImg：左边的图片数据。
    # rightImg：右边的图片数据。
    # roi：一个包含 ROI 信息的元组，顺序为 (top, bottom, left, right, extend)。（四个顶点加边长）
    # leftTitle：左边图片的标题。
    # rightTitle：右边图片的标题。
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(leftImg)
    plt.title(leftTitle)

    plt.subplot(2, 2, 2)
    plt.imshow(rightImg)
    plt.title(rightTitle)

    ax = plt.subplot(2, 2, 3)
    top, bottom, left, right, extend = roi
    rect = plt.Rectangle([left, top], right - left, bottom - top, edgecolor='Red', facecolor='None')
    ax.add_patch(rect)
    plt.imshow(leftImg)
    plt.ylim(bottom + extend, top - extend)
    plt.xlim(left - extend, right + extend)
    plt.title("ROI of " + leftTitle)

    ax = plt.subplot(2, 2, 4)
    top, bottom, left, right, extend = roi
    rect = plt.Rectangle([left, top], right - left, bottom - top, edgecolor='Red', facecolor='None')
    ax.add_patch(rect)
    plt.imshow(rightImg)
    plt.ylim(bottom + extend, top - extend)
    plt.xlim(left - extend, right + extend)
    plt.title("ROI of " + rightTitle)

    plt.show()

# 对浮点型数组中的元素值进行限制，确保它们在一个特定的范围内的函数，返回把元素范围改好过后的图形数组
def castFloatArray(imgIn, lowLimit=-1, highLimit=1):
    #三个参数，接受的图像数组，元素值默认下限-1，上限为1
    imgOut = imgIn.copy()#避免直接修改原始数组
    if imgIn.dtype == np.float32 or imgIn.dtype == np.float64:
        imgOut[imgOut > highLimit] = highLimit
        imgOut[imgOut < lowLimit] = lowLimit
    return imgOut

# 对输入的数组进行归一化处理，使其元素值位于 [0, 1] 区间内。
def normFloatArray(imgIn):
    imgOut = imgIn.copy()# 避免直接修改原始数组
    if imgIn.max() == imgIn.min():
        imgOut = np.zeros(imgIn.shape)
    elif len(imgIn.shape) == 2:# 检查 imgIn 是否为二维数组，这通常意味着它是一幅灰度图像。
        imgOut = (imgOut - imgOut.min()) / (imgOut.max() - imgOut.min())
    elif len(imgIn.shape) == 3:# 检查 imgIn 是否为三维数组，这通常意味着它是一幅彩色图像
        for c in range(3):
            imgOut[:, :, c] = (imgOut[:, :, c] - imgOut[:, :, c].min()) / (
                        imgOut[:, :, c].max() - imgOut[:, :, c].min())
    return imgOut.astype(np.float32)# 最后，将归一化后的数组 imgOut 的数据类型转换为 np.float32，然后返回。

def yiq2rgbUint(imgIn):
    imgOut = (255 * normFloatArray(color.yiq2rgb(castFloatArray(imgIn)))).astype(np.uint8)
    return imgOut
# 首先调用 castFloatArray 函数将输入的 imgIn 转换为浮点型数组，并确保其值在合理范围内。
# 然后调用 color.yiq2rgb 函数将 YIQ 彩色空间的图像转换为 RGB 彩色空间。
# 接着调用 normFloatArray 函数对 RGB 图像进行归一化处理，确保其值在 [0, 1] 范围内。
# 最后，将归一化后的图像乘以 255 并转换为 np.uint8 类型，得到适合显示的整数像素值。

def yiq2bgrUint(imgIn):
    imgOut = (255 * normFloatArray(color.yiq2rgb(castFloatArray(imgIn)))).astype(np.uint8)
    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_RGB2BGR)
    return imgOut
# 此函数与 yiq2rgbUint 类似，但在最后使用 cv2.cvtColor 函数将 RGB 图像转换为 BGR 格式，这是 OpenCV 默认的色彩空间。

def yiq2rgbFloat(imgIn):
    imgOut = normFloatArray(color.yiq2rgb(castFloatArray(imgIn)))
    return imgOut
# 此函数与 yiq2rgbUint 类似，但是最终返回的是浮点型数组，没有乘以 255 和转换为 uint8 类型，因此输出的图像数据类型仍然是浮点型。

def yiq2bgrFloat(imgIn):
    tmp = imgOut = normFloatArray(color.yiq2rgb(castFloatArray(imgIn)))
    imgOut = tmp.copy()
    imgOut[:, :, 0] = tmp[:, :, 2]
    imgOut[:, :, 1] = tmp[:, :, 1]
    imgOut[:, :, 2] = tmp[:, :, 0]
    return imgOut
# 此函数与 yiq2rgbFloat 类似，但首先将 RGB 图像转换为 BGR 格式，然后交换 RGB 三个通道的值。

# 比较两个单独的视频帧，并在图像上标注出一个特定的 ROI（感兴趣区域）
def compareFrames(frames, frameId1, frameId2, roi):
    # 接受四个参数：
    # frames：一个包含多个视频帧的数组，通常是一个四维数组，其中第一维代表帧的编号。
    # frameId1：要比较的第一个帧的编号。
    # frameId2：要比较的第二个帧的编号。
    # roi：一个包含
    # ROI
    # 信息的元组，用于在图像上绘制矩形。
    frame1 = frames[frameId1, :, :, :]
    frame2 = frames[frameId2, :, :, :]
    frame1rgb = yiq2rgbFloat(frame1)#转换颜色空间
    frame2rgb = yiq2rgbFloat(frame2)
    compareImg(frame1rgb, frame2rgb, roi, 'Frame %d' % frameId1, 'Frame %d' % frameId2)
    #再使用前面定义的两张图像的对比函数

# 定义用于绘制视频中某一点在时间轴上的信号变化，特别是针对 YIQ 色彩空间的视频帧。
def draw1DTemporalSignal(frames, ypos, xpos, fps):
    # 接受四个参数：
    # frames：一个四维数组，包含了视频的多个帧，形状为( frameNum（帧数）、H（高度）、W（宽度）, （通道数）)。
    # ypos：要绘制的时间序列信号所在像素点的y坐标。
    # xpos：要绘制的时间序列信号所在像素点的x坐标。
    # fps：视频的帧率，单位为帧每秒。
    frameNum, H, W, chNum = frames.shape
    tlist = [t*1000/fps for t in range(frameNum)]
    #创建一个列表 tlist，包含从 0 到 frameNum-1 的时间戳，单位为毫秒。每个时间戳计算方式为 t*1000/fps，其中 t 表示帧的编号。
    plt.figure(figsize=(20,5))#创建一个画布，大小为20x5英寸。
    chName=['y', 'i', 'q']
    for c in range(chNum):#遍历 frames 的所有通道
        plt.subplot(1,3,c+1)#创建一个子图，子图数量为 1x3，当前子图编号为 c+1。
        plt.plot(tlist, frames[:,ypos, xpos, c], 'r-')
        #绘制时间序列信号，横坐标为 tlist，纵坐标为 frames 中指定像素点在当前通道 c 上的值。线条颜色为红色
        plt.title("%s channel pixel value change for point(%d, %d)"%(chName[c], ypos, xpos))
        #设置子图的标题，描述当前通道名称以及像素点的位置。
        plt.xlabel('Time(ms)')
        #设置横坐标的标签为“时间（毫秒）”。
        plt.ylabel('Pixel Value')
        #设置纵坐标的标签为“像素值”。
    plt.show()#显示所有创建的子图，即每个通道的时间序列信号图。

# 定义了一个名为，用于构建图像的拉普拉斯金字塔
def buildLaplacianPyramid(imgIn, maxLevel):# 两个参数：原始图像，金字塔最大层数
    currentLayer = 0# 初始化当前层变量 currentLayer 为 0，用于追踪当前构建的金字塔层级。
    imgPyramid = []# 创建一个空列表 imgPyramid，用于存储金字塔的每一层图像。
    curSrc = imgIn.astype(np.float32)# 将输入图像 imgIn 转换为 float32 类型，并赋值给 curSrc，作为金字塔构建的起始图像。
    while (currentLayer < maxLevel - 1):
        # 开始一个循环，直到当前层达到，maxLevel - 1。实际构建的层数可能少于，maxLevel，具体取决于图像尺寸。
        curH, curW = curSrc.shape[0:2]# 获取当前图像的高度 curH 和宽度 curW。
        if curH < 4 or curW < 4:
            break
        # 检查当前图像的尺寸是否小于 4x4。如果小于，则退出循环，防止后续操作导致图像失真。
        currentLayer += 1# 增加当前层的计数。
        imgBlur = cv2.GaussianBlur(curSrc, ksize=(0, 0), sigmaX=3)
        # 使用高斯模糊滤波器对当前图像 curSrc 进行模糊处理，sigmaX=3 控制模糊程度。
        imgBlurDown = cv2.resize(imgBlur, ((curW + 1) // 2, (curH + 1) // 2))
        # 将模糊后的图像 imgBlur 缩小一半，尺寸变为 (curW + 1) // 2 和 (curH + 1) // 2。
        imgBlurUp = cv2.resize(imgBlurDown, (curW, curH))
        # 将缩小后的图像 imgBlurDown 放大回原尺寸。
        imgBlurUp = cv2.GaussianBlur(imgBlurUp, ksize=(0, 0), sigmaX=3)
        # 再次对放大后的图像 imgBlurUp 进行高斯模糊处理。
        imgDiff = curSrc - imgBlurUp
        # 计算原始图像 curSrc 与模糊放大的图像 imgBlurUp 之间的差值，得到当前层的拉普拉斯图像。
        imgPyramid.append(imgDiff)
        # 将当前层的拉普拉斯图像添加到金字塔列表 imgPyramid 中。
        curSrc = imgBlurDown
        # 更新 curSrc 为缩小后的模糊图像 imgBlurDown，准备进入下一层的构建。

    imgPyramid.append(curSrc)# 循环结束后，将最后一层（最底层）的图像添加到金字塔列表 imgPyramid 中。
    return imgPyramid # 返回构建的金字塔列表 imgPyramid。

# 从拉普拉斯金字塔（Laplacian Pyramid）重建图像
def recreateImgsFromLapPyr(imgPyramid):
    #当循环结束时，curSrc 将包含重建后的完整图像，函数返回这个结果。
    #通过这个函数，你可以从拉普拉斯金字塔中重建原始图像，或者根据需要调整某些层的细节后再进行重建，
    #从而实现图像的增强或融合。

    layerNum = len(imgPyramid)
    curSrc = imgPyramid[-1].copy()
    for l in np.arange(layerNum - 2, -1, -1):
        imgUp = cv2.resize(curSrc, (imgPyramid[l].shape[1], imgPyramid[l].shape[0]))
        imgBlurUp = cv2.GaussianBlur(imgUp, ksize=(0, 0), sigmaX=3)
        curSrc = imgBlurUp + imgPyramid[l]

    return (curSrc)

# 用于测试拉普拉斯金字塔的构建和图像重建过程。
# 函数接收一个图像、金字塔的最大层数以及一个感兴趣区域（ROI），
# 并使用这些参数来验证图像金字塔的正确性和重建效果。
def testImgPyramid(image, maxLevel, roi):
    # 接受三个参数：
    # image：待处理的原始图像。
    # maxLevel：构建拉普拉斯金字塔的最大层数。
    # roi：感兴趣区域的信息，用于在比较图像时突出显示。
    imgPyramid = buildLaplacianPyramid(image, maxLevel)
    #
    recreateImg = recreateImgsFromLapPyr(imgPyramid)
    recreateImg=yiq2rgbFloat(recreateImg)
    compareImg(yiq2rgbFloat(image), recreateImg, roi, "Original YIQ Frame 7", "Recreate YIQ Frame 7")
    # 通过这个函数，你可以评估拉普拉斯金字塔的构建和图像重建过程的效果，
    # 特别是在视觉上检查重建图像是否准确地恢复了原始图像的细节。

#展示图像的拉普拉斯金字塔
def testShowPyramid(image, maxLevel):#输入图像和拉普拉斯金字塔最大层数
    rows, cols = image.shape[0:2]
    imgPyramid = buildLaplacianPyramid(image, maxLevel)
    composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.float32)
    composite_image[:rows, :cols, :] = normFloatArray(imgPyramid[0])
    i_row = 0
    for p in imgPyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = normFloatArray(p)
        i_row += n_rows
#显示拉普拉斯金字塔
    plt.figure(figsize=(15,15))
    plt.title("Laplacian Pyramid Show for Frame 7")
    plt.imshow(composite_image)
    plt.show()

#构建视频帧的拉普拉斯金字塔
def buildVideoLapPyr(frames, maxLevel):
#frames: 视频帧的列表，每个元素是一幅图像，maxLevel: 拉普拉斯金字塔的最大层数。
    pyr0 = buildLaplacianPyramid(frames[0], maxLevel)#构建第一帧的拉普拉斯金字塔
    realMaxLevel = len(pyr0)#确定实际最大层数

    resultList = []
    for i in range(realMaxLevel):
        curPyr = np.zeros([len(frames)] + list(pyr0[i].shape), dtype=np.float32)
        #创建一个三维数组，其形状为 [帧数, 当前层的高度, 当前层的宽度]
        resultList.append(curPyr)

    for fn in range(len(frames)):#遍历金字塔的每一层
        pyrOfFrame = buildLaplacianPyramid(frames[fn], maxLevel)
        for i in range(realMaxLevel):
            resultList[i][fn] = pyrOfFrame[i]
    #函数结束时返回一个列表，其中每个元素是一个三维数组，表示视频帧在金字塔某一层上的表现。

    return resultList

#从拉普拉斯金字塔重构视频
def recreateVideoFromLapPyr(pyrVideo):
    #其中每个元素是一个三维数组，代表视频帧在金字塔各层的表现
    maxLevel = len(pyrVideo)#计算金字塔的层数，即 pyrVideo 列表的长度。
    #获取视频的基本尺寸，获取视频的帧数 (fNumber)、高度 (H)、宽度 (W) 和通道数 (chNum)
    fNumber, H, W, chNum = pyrVideo[0].shape
    videoResult = np.zeros(pyrVideo[0].shape, dtype=np.float32)
    #创建一个全零的 numpy 数组，形状与金字塔的第一层相同，数据类型为 float32，作为重构后的视频。
    for fn in range(videoResult.shape[0]):
        framePyr = [pyrVideo[i][fn] for i in range(maxLevel)]
        #为当前帧创建一个金字塔列表，其中包含了该帧在金字塔每一层的表示
        videoResult[fn] = recreateImgsFromLapPyr(framePyr)
        #传入当前帧的金字塔列表，以重构出原始帧

    return videoResult#函数执行完毕后，返回重构后的视频 videoResult

#可视化视频中特定点在时间序列上的信号变化
def showTemporalSignal(frames, fps, pyrVideo, layer, ypos, xpos, keyword=""):
    #接收多个参数：
    # frames: 视频的帧列表。
    # fps: 视频的帧率（每秒帧数）。
    # pyrVideo: 视频的拉普拉斯金字塔表示。
    # layer: 要分析的金字塔层。
    # ypos, xpos: 视频帧中感兴趣点的坐标。
    # keyword: 可选的关键词，用于标题中描述金字塔的类型。
    tlist = [t * 1000 / fps for t in range(pyrVideo[layer].shape[0])]
    #根据视频的帧率和金字塔层的帧数，计算时间戳列表，单位为毫秒。
    plt.figure(figsize=(30, 5))#创建一个大小为 30x5 英寸的绘图窗口
    chName = ['Y', 'I', 'Q']#假设视频有 YIQ 三个颜色通道
    chNum = len(chName)#获取通道数量
    ax = plt.subplot(1, 4, 1)#创建一个 1 行 4 列的子图布局中的第一个子图
    plt.title("Frame 0 Layer %d of the %s Pyramid" % (layer, keyword))
    #设置子图的标题，描述当前显示的是第 0 帧的金字塔的哪一层
    frame2show = normFloatArray(pyrVideo[layer][0])
    #从金字塔中选取第 0 帧的指定层，规范化后显示
    markRadius = 10
    #创建一个红色的矩形，用于在图像上标记感兴趣点的位置。这里使用了位移运算符 >> 来调整坐标和尺寸，考虑到金字塔每一层的分辨率不同
    rect = plt.Rectangle([(xpos - markRadius) >> layer, (ypos - markRadius) >> layer], markRadius * 2 >> layer,
                         markRadius * 2 >> layer, edgecolor='Red', facecolor='red')
    plt.imshow(frame2show)
    ax.add_patch(rect)#添加标记区域到图像
    for c in range(chNum):#遍历视频的每个颜色通道
        plt.subplot(1, 4, c + 2)#创建子图布局中的下一个子图
        plt.plot(tlist, pyrVideo[layer][:, ypos >> layer, xpos >> layer, c], 'r-')
        #绘制时间序列信号，显示感兴趣点在当前层和通道上的像素值随时间的变化
        plt.title("Layer %d:%s channel value for point(%d, %d)" % (layer, chName[c], ypos, xpos))
        #设置子图的标题，描述显示的是哪一层、哪个通道的信号
        plt.xlabel('Time(ms)')
        #设置 x 轴和 y 轴标签
        plt.ylabel('Pixel Value')
    plt.show()#打开图形界面并显示所有子图。

#功能是在指定轴上对输入数据应用Butterworth带通滤波器
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    #函数接收以下参数：
    # data：待过滤的数据数组。
    # lowcut：带通滤波器的低截止频率。
    # highcut：带通滤波器的高截止频率。
    # fs：数据的采样频率。
    # order：Butterworth滤波器的阶数，默认为5。
    # axis：在数据数组上应用线性滤波的轴，默认为0轴。
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega#截止频率被归一化，即除以采样频率的一半(omega)。
    b, a = signal.butter(order, [low, high], btype='band')
    #使用scipy.signal模块中的signal.butter函数计算滤波器的系数(b 和 a)。btype='band'参数表明要设计的是带通滤波器
    y = signal.lfilter(b, a, data, axis=axis)#使用signal.lfilter函数将滤波器应用于数据。axis参数指定了滤波应该沿数据数组的哪一轴进行。
    return y#函数返回过滤后的数据

#对多维数据（如视频或音频信号）应用理想带通滤波器，以保留特定频率范围内的信号成分。
def temporal_ideal_filter(tensor, low, high, fps, axis=0):
    # 接收五个参数：
    # tensor：多维数据数组，例如视频帧或音频信号。
    # low：滤波器的低截止频率。
    # high：滤波器的高截止频率。
    # fps：数据的采样率，对于视频而言是帧率。
    # axis：沿着哪个轴应用FFT变换，默认为0轴。
    fft = fftpack.fft(tensor, axis=axis)
    #使用 fftpack.fft 函数对 tensor 沿着指定轴进行快速傅里叶变换（FFT），得到频域表示
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    #使用 fftpack.fftfreq 函数计算FFT结果对应的频率轴。d 参数是采样间隔，等于1/fps
    bound_low = (np.abs(frequencies - low)).argmin()
    #计算频率轴上最接近 low 的频率值的索引位置。
    bound_high = (np.abs(frequencies - high)).argmin()
    #同上，计算频率轴上最接近 high 的频率值的索引位置。
    if (bound_low == bound_high) and (bound_high < len(fft) - 1):
        bound_high += 1
        #处理边界情况，如果 bound_low 和 bound_high 相同，并且 bound_high 不是倒数第二个元素，则增加 bound_high 的值，确保区间不为空。
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    #将低于 bound_low 和高于 bound_high 的频率成分置零，实现理想带通滤波
    iff = fftpack.ifft(fft, axis=axis)#返回IFFT结果的绝对值，这是因为IFFT的结果可能是复数，取绝对值可以得到实际的信号强度。

    return np.abs(iff)
#这个函数的主要作用是对输入的多维数据应用一个理想带通滤波器，只保留特定频率范围内的信号成分，而其他频率成分则被完全去除。

#用于对视频金字塔结构中的每一层应用理想带通滤波器。
def idealFilterForVideoPyr(videoPyr, low, high, fps, roi=None):
    #接收四个参数和一个可选参数：
    # videoPyr：视频金字塔，通常是一个列表，其中每个元素代表一个分辨率级别的视频帧。
    # low：滤波器的低截止频率。
    # high：滤波器的高截止频率。
    # fps：视频的帧率，即采样率。
    # roi：可选参数，如果指定，仅对视频帧的感兴趣区域（Region of Interest, ROI）应用滤波。但注释中指出，这部分功能尚未实现。
    resultVideoPyr = []#创建一个空列表，用于存储经过滤波处理后的视频金字塔各层。
    for layer in tqdm(range(len(videoPyr)), desc="Filtering video pyramid layers"):#循环处理视频金字塔的每一层：
        filteredPyr = temporal_ideal_filter(videoPyr[layer], low, high, fps, axis=0)#应用理想带通滤波器：
        resultVideoPyr.append(filteredPyr)#将当前层的滤波结果添加到结果列表中。

    return resultVideoPyr#函数返回一个列表，其中包含了视频金字塔每一层经过滤波处理后的结果。
#这个函数的主要目的是对视频金字塔结构中的每一层分别应用理想带通滤波器，从而在不同的分辨率级别上保留特定频率范围内的信号成分，
#注意，函数的文档字符串中提到的ROI功能尚未实现，这意味着目前该函数会处理整个视频帧，而不会限制于特定的感兴趣区域。

#对视频金字塔结构中的每一层应用Butterworth带通滤波器。
def buttFilterForVideoPyr(videoPyr, low, high, fps, roi=None):
    resultVideoPyr = []
    for layer in range(len(videoPyr)):
        filteredPyr = butter_bandpass_filter(videoPyr[layer], low, high, fps, order=1, axis=0)
        resultVideoPyr.append(filteredPyr)
#函数返回一个列表，其中包含了视频金字塔每一层经过Butterworth带通滤波处理后的结果。
    return resultVideoPyr
#对视频金字塔结构中的每一层分别应用Butterworth带通滤波器，从而在不同的分辨率级别上保留特定频率范围内的信号成分，同时平滑地过渡到被滤除的频率范
#注意，函数的文档字符串中提到的ROI功能尚未实现，这意味着目前该函数会处理整个视频帧，而不会限制于特定的感兴趣区域。


def temporal_ideal_filter(tensor, low, high, fps, axis=0):
    # 使用 fftpack.fft 函数对 tensor 沿着指定轴进行快速傅里叶变换（FFT），得到频域表示
    fft = fftpack.fft(tensor, axis=axis)
    # 使用 fftpack.fftfreq 函数计算FFT结果对应的频率轴。d 参数是采样间隔，等于1/fps
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    # 计算频率轴上最接近 low 和 high 的频率值的索引位置。
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()

    if (bound_low == bound_high) and (bound_high < len(fft) - 1):
        bound_high += 1

    # 将低于 bound_low 和高于 bound_high 的频率成分置零，实现理想带通滤波
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0

    # 返回IFFT结果的绝对值，这是因为IFFT的结果可能是复数，取绝对值可以得到实际的信号强度。
    iff = fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

# def idealFilterForVideoPyr_new(videoPyr, low, high, fps, roi=None, chunk_size=30):
#     #chunk_size越小，空间占用越小，计算时间越长。越大则反之，
#     resultVideoPyr = []
#     for layer in range(len(videoPyr)):
#         layer_pyr = videoPyr[layer]
#         filtered_layer_pyr = np.zeros_like(layer_pyr)
#         for i in range(0, len(layer_pyr), chunk_size):
#             chunk = layer_pyr[i:i+chunk_size]
#             filtered_chunk = temporal_ideal_filter(chunk, low, high, fps, axis=0)
#             filtered_layer_pyr[i:i+chunk_size] = filtered_chunk
#         resultVideoPyr.append(filtered_layer_pyr)
#     return resultVideoPyr

def idealFilterForVideoPyr_new(videoPyr, low, high, fps, roi=None, chunk_size=30, overlap=10):
    resultVideoPyr = []
    for layer in range(len(videoPyr)):
        layer_pyr = videoPyr[layer]
        filtered_layer_pyr = np.zeros_like(layer_pyr)
        step_size = chunk_size - overlap  # 步长：块大小减去重叠部分

        for i in range(0, len(layer_pyr), step_size):
            chunk = layer_pyr[i:i+chunk_size]
            if chunk.shape[0] < chunk_size:
                chunk = np.pad(chunk, ((0, chunk_size - chunk.shape[0]), (0, 0), (0, 0), (0, 0)), mode='edge')
            filtered_chunk = temporal_ideal_filter(chunk, low, high, fps, axis=0)
            filtered_layer_pyr[i:i+chunk_size] = filtered_chunk[:min(chunk_size, len(layer_pyr) - i)]

        resultVideoPyr.append(filtered_layer_pyr)
    return resultVideoPyr

def amplifyTemporalColorSignal(originalPyr, filteredVideoPyr, alpha, chromAttenuation):
    amplifiedPyr = []

    for layer in tqdm(range(len(filteredVideoPyr)), desc="Amplifying color signal"):
        tensor = originalPyr[layer].copy()
        if layer == len(filteredVideoPyr) - 1:
            tensor[:, :, :, 0] += filteredVideoPyr[layer][:, :, :, 0] * alpha
            tensor[:, :, :, 1] += filteredVideoPyr[layer][:, :, :, 1] * alpha * chromAttenuation
            tensor[:, :, :, 2] += filteredVideoPyr[layer][:, :, :, 2] * alpha * chromAttenuation

        amplifiedPyr.append(tensor)

    return amplifiedPyr


def amplifyTemporalMotionSignal(originalPyr, filteredVideoPyr, alpha, lambdaC, chromAttenuation):
    amplifiedPyr = []

    delta = lambdaC / 8 / (1 + alpha)
    frameCount, H, W, chCount = originalPyr[0].shape
    lamb = np.sqrt(H ** 2 + W ** 2) / 3.0

    for layer in tqdm(range(len(filteredVideoPyr)), desc="Amplifying motion signal"):
        tensor = originalPyr[layer].copy()
        if layer == len(filteredVideoPyr) - 1 or layer == 0:
            currAlpha = 0
        elif currAlpha > alpha:
            currAlpha = alpha
        else:
            currAlpha = lamb / delta / 8 - 1

        tensor[:, :, :, 0] += filteredVideoPyr[layer][:, :, :, 0] * currAlpha
        tensor[:, :, :, 1] += filteredVideoPyr[layer][:, :, :, 1] * currAlpha * chromAttenuation
        tensor[:, :, :, 2] += filteredVideoPyr[layer][:, :, :, 2] * currAlpha * chromAttenuation

        amplifiedPyr.append(tensor)
        lamb /= 2

    return amplifiedPyr

#将一系列帧保存为视频文件。
# def saveFramesToVideo(frames, videoPath):#接受两个参数frames 和 videoPath。
#     # frames 是一个包含图像帧的数组，videoPath 是输出视频文件的路径。
#     #  使用 OpenCV 的 VideoWriter_fourcc 函数创建一个 FourCC 编码器。
#     #  这里使用的是 'a', 'v', 'c', '1'，通常表示 H.264 编码
#     fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
#     [height, width] = frames[0].shape[0:2]#获取第一个帧的高度和宽度。frames[0].shape[0:2] 返回一个元组，其中包含了帧的高度和宽度。
#     #创建一个 cv2.VideoWriter 对象，用于将帧写入视频
#     writer = cv2.VideoWriter(videoPath, fourcc, 30, (width, height), 1)
#     #参数包括：
#     # 输出视频的路径 (videoPath)
#     # 四字符代码 (fourcc)，用于指定视频编解码器
#     # 帧率 (30)，即每秒帧数
#     # 视频的尺寸 ((width, height))
#     # 是否为彩色视频 (1 表示是，0 表示不是)
#     for i in range(frames.shape[0]):#遍历所有帧。对于每一帧
#         #调用 yiq2bgrUint 函数将帧从 YIQ 色彩空间转换到 BGR 色彩空间，并确保数据类型适合写入视频。
#         frameBGR = yiq2bgrUint(frames[i])
#         #使用 writer.write() 方法将转换后的帧写入视频
#         writer.write(frameBGR)
#     writer.release()
#
# #功能是在原始帧上叠加或替换指定区域（Region of Interest，ROI）的帧，然后将处理后的帧序列保存为视频文件。
# def saveFramesToVideoROI(orgFrames, recreateFrames, videoPath, roi=None):
#     # 四个参数：
#     # orgFrames: 原始帧序列
#     # recreateFrames: 用于替换或叠加的帧序列
#     # videoPath: 输出视频的路径
#     # roi: 可选参数，表示要操作的区域（左上角和右下角坐标）
#     fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
#     [height, width] = orgFrames[0].shape[0:2]# 获取原始帧的尺寸（高度和宽度）。
#     if roi is None:
#         #根据 roi 参数设置区域边界。如果 roi 未提供，则默认为整个帧；否则，从 roi 中提取边界值。
#         top = 0
#         bottom = height
#         left = 0
#         right = width
#     else:
#         [top, bottom, left, right] = roi
#     #创建 cv2.VideoWriter 对象，初始化视频写入器。
#     writer = cv2.VideoWriter(videoPath, fourcc, 30, (width, height), 1)
#     for i in range(recreateFrames.shape[0]):
#         #遍历 recreateFrames 序列中的每个帧：
#         # 将当前帧从 YIQ 色彩空间转换到 BGR 色彩空间，存储在 recreateFramesBGR 中。
#         # 同样将原始帧转换到 BGR 色彩空间，存储在 saveFrame 中。
#         # 使用 NumPy 的切片操作将 recreateFramesBGR 的 ROI 替换到 saveFrame 的相应位置。
#         # 将处理后的 saveFrame 写入视频。
#         recreateFramesBGR = yiq2bgrUint(recreateFrames[i])
#         saveFrame = yiq2bgrUint(orgFrames[i])
#         saveFrame[top:bottom, left:right] = recreateFramesBGR[top:bottom, left:right]
#         writer.write(saveFrame)
#     writer.release()


# 更新 saveFramesToVideo 和 saveFramesToVideoROI 函数，添加进度条
def saveFramesToVideo(frames, videoPath, fps):
    from tqdm import tqdm  # 添加进度条库
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    [height, width] = frames[0].shape[0:2]
    writer = cv2.VideoWriter(videoPath, fourcc, fps, (width, height), 1)
    for i in tqdm(range(frames.shape[0]), desc="Saving frames to video"):  # 添加进度条
        frameBGR = yiq2bgrUint(frames[i])
        writer.write(frameBGR)
    writer.release()

def saveFramesToVideoROI(orgFrames, recreateFrames, videoPath, fps, roi=None):
    from tqdm import tqdm  # 添加进度条库
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    [height, width] = orgFrames[0].shape[0:2]
    if roi is None:
        top = 0
        bottom = height
        left = 0
        right = width
    else:
        [top, bottom, left, right] = roi
    writer = cv2.VideoWriter(videoPath, fourcc, fps, (width, height), 1)
    for i in tqdm(range(recreateFrames.shape[0]), desc="Saving frames to video with ROI"):  # 添加进度条
        recreateFramesBGR = yiq2bgrUint(recreateFrames[i])
        saveFrame = yiq2bgrUint(orgFrames[i])
        saveFrame[top:bottom, left:right] = recreateFramesBGR[top:bottom, left:right]
        writer.write(saveFrame)
    writer.release()

#主要功能是对视频帧序列进行颜色信号的频率滤波和放大，以增强或减弱特定频率范围内的视觉效果。
def emvCoreColor(frames, fps, maxLevel, freqLow, freqHigh, alpha, chromAttenuation, method="ideal"):
    #接收以下参数：
    # frames: 输入的视频帧序列
    # fps: 视频的帧率
    # maxLevel: 拉普拉斯金字塔的最大层数
    # freqLow 和 freqHigh: 频率滤波的低频和高频界限
    # alpha: 时间信号放大的系数
    # chromAttenuation: 色度衰减系数
    # method: 滤波方法，可以是 "ideal" 或 "butt"，分别对应理想滤波和巴特沃斯滤波，默认为 "ideal"
    roi = None
    chunk_size = 30
    pyrVideo_ = buildVideoLapPyr(frames, maxLevel)#使用 buildVideoLapPyr 函数构建拉普拉斯金字塔，这是一种多分辨率表示，用于分解视频帧的不同频率成分。
    #根据 method 参数选择滤波方法。这两种滤波器都会根据给定的频率界限对拉普拉斯金字塔中的视频帧进行滤波。
    if method == "ideal":
        filteredVideoPyr = idealFilterForVideoPyr(pyrVideo_, freqLow, freqHigh, fps)
        #filteredVideoPyr = idealFilterForVideoPyr_new(pyrVideo_, freqLow, freqHigh, fps, roi, chunk_size, 10)
    elif method == "butt":
        filteredVideoPyr = buttFilterForVideoPyr(pyrVideo_, freqLow, freqHigh, fps)
    #对经过滤波的拉普拉斯金字塔进行时间信号放大，这一步会根据 alpha 和 chromAttenuation 参数调整信号强度
    amplifiedPyr = amplifyTemporalColorSignal(pyrVideo_, filteredVideoPyr, alpha, chromAttenuation)
    #从放大后的拉普拉斯金字塔重建视频帧序列
    recreateFrames = recreateVideoFromLapPyr(amplifiedPyr)

    return recreateFrames

#作用是处理视频帧序列，通过频率滤波和信号放大来增强视频中的运动信号。
def emvCoreMotion(frames, fps, maxLevel, freqLow, freqHigh, alpha, lambdaC, chromAttenuation, method="ideal"):
    #frames: 输入的视频帧序列
    # fps: 视频的帧率
    # maxLevel: 构建拉普拉斯金字塔的最大层数
    # freqLow, freqHigh: 频率滤波的低频和高频界限
    # alpha: 时间信号放大的系数
    # lambdaC: 用于运动信号放大的额外参数
    # chromAttenuation: 色度衰减系数
    # method: 滤波方法，可以是 "ideal" 或 "butt"，分别对应理想滤波和巴特沃斯滤波
    pyrVideo_ = buildVideoLapPyr(frames, maxLevel)
    if method == "ideal":
        filteredVideoPyr = idealFilterForVideoPyr(pyrVideo_, freqLow, freqHigh, fps)
    elif method == "butt":
        filteredVideoPyr = buttFilterForVideoPyr(pyrVideo_, freqLow, freqHigh, fps)

    amplifiedPyr = amplifyTemporalMotionSignal(pyrVideo_, filteredVideoPyr, alpha, lambdaC, chromAttenuation)
    recreateFrames = recreateVideoFromLapPyr(amplifiedPyr)

    return recreateFrames

#定义了 emv 函数，这是一个高级接口，用于执行视频信号增强，具体是颜色信号增强或运动信号增强，然后将结果保存为新视频。
# def emv(inputVideoPath, outputVideoPath, maxLevel, freqLow, freqHigh, alpha, chromAttenuation, startFrameNumber,
#         endFrameNumber, lambdaC=-1, app="color", method="ideal", roi=None):
#     #inputVideoPath: 输入视频文件的路径
#     # outputVideoPath: 输出视频文件的路径
#     # maxLevel: 构建拉普拉斯金字塔的最大层数
#     # freqLow, freqHigh: 频率滤波的低频和高频界限
#     # alpha: 时间信号放大的系数
#     # chromAttenuation: 色度衰减系数
#     # startFrameNumber, endFrameNumber: 处理视频帧的起始和结束编号
#     # lambdaC: 运动信号增强的额外参数（默认为 -1）
#     # app: 应用模式，可以是 "color" 或 "motion"，分别对应颜色信号增强和运动信号增强
#     # method: 滤波方法，可以是 "ideal" 或 "butt"，分别对应理想滤波和巴特沃斯滤波
#     # roi: 感兴趣区域（Region of Interest），用于指定视频中要处理的部分，默认为 None，意味着处理整个视频
#     fps, frames = getVideoFrames(inputVideoPath, startFrameNumber, endFrameNumber)
#     #使用 getVideoFrames 函数从输入视频中获取指定范围内的帧及其帧率（fps）。startFrameNumber 和 endFrameNumber 参数用于控制读取视频的开始和结束帧。
#     if app == "color":
#         recreateFrames = emvCoreColor(frames, fps, maxLevel, freqLow, freqHigh, alpha, chromAttenuation, method)
#     elif app == "motion":
#         recreateFrames = emvCoreMotion(frames, fps, maxLevel, freqLow, freqHigh, alpha, lambdaC, chromAttenuation,
#                                        method)
#     saveFramesToVideoROI(frames, recreateFrames, outputVideoPath, roi)
#     return

# 修改主要函数
def emv(inputVideoPath, outputVideoPath, maxLevel, freqLow, freqHigh, alpha, chromAttenuation,
        startFrameNumber, endFrameNumber, lambdaC=-1, app="color", method="ideal", roi=None):
    total_start = time.time()
    logging.info(f"\nStarting video processing: {inputVideoPath}")
    logging.info(f"Parameters: maxLevel={maxLevel}, freqLow={freqLow}, freqHigh={freqHigh}, alpha={alpha}")

    # 读取视频
    io_start = time.time()
    fps, frames = getVideoFrames(inputVideoPath, startFrameNumber, endFrameNumber)
    logging.info(f"Video reading time: {time.time() - io_start:.2f} seconds")

    # 核心处理
    process_start = time.time()
    if app == "color":
        recreateFrames = emvCoreColor(frames, fps, maxLevel, freqLow, freqHigh, alpha, chromAttenuation, method)
    elif app == "motion":
        recreateFrames = emvCoreMotion(frames, fps, maxLevel, freqLow, freqHigh, alpha, lambdaC, chromAttenuation,
                                       method)
    processing_time = time.time() - process_start
    logging.info(f"Core processing time: {processing_time:.2f} seconds")

    # 保存结果
    save_start = time.time()
    saveFramesToVideoROI(frames, recreateFrames, outputVideoPath, fps, roi)
    logging.info(f"Video saving time: {time.time() - save_start:.2f} seconds")

    # 记录总体性能
    total_duration, memory_usage, cpu_usage = log_performance("Total Process", total_start)

    return recreateFrames


# 在调用处添加计时
if __name__ == "__main__":
    emv('E:/amplify/input/hx032.mp4',
        'E:/Ulcer_classify/input/hx032_rebuild_CPU.mp4',
        6, 0.83 / 2, 1.0 / 2, 100.0, 1, 0, -1, -1,
        "color", "ideal")