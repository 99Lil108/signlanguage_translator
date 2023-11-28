import time

import cv2

# 打开视频文件
video_capture = cv2.VideoCapture('cutebb.mp4')

# 初始化计数器
frame_count = 0

while True:
    # 读取视频帧
    ret, frame = video_capture.read()

    # 如果视频帧读取失败，则退出循环
    if not ret:
        break

    # 每隔12帧显示一帧
    if frame_count == 0:
        cv2.imshow('Video', frame)

    frame_count += 1
    frame_count %= 12

# 释放资源
video_capture.release()
cv2.destroyAllWindows()
