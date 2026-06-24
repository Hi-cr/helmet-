!pip install ultralytics

from ultralytics import YOLO

!pip install roboflow

!pip install

from roboflow import Roboflow
rf = Roboflow(api_key="vE5YP9MAsVrXS5ZhP53E")
project = rf.workspace("construction-helmet-sthqm").project("construction-helmet-detection-dxplv")
version = project.version(1)
dataset = version.download("yolov8")

print(dataset.location)

model = YOLO("yolov8n.pt")

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=10,#會調整的超參數
    imgsz=640,
    batch=16

from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("runs/detect/train/results.png")

plt.figure(figsize=(12,8))
plt.imshow(img)
plt.axis("off")
plt.show()

import pandas as pd

df = pd.read_csv("runs/detect/train/results.csv")

print(df.columns)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/detect/train/results.csv")

plt.figure(figsize=(8,5))
plt.plot(df['epoch'],
         df['metrics/mAP50(B)'],
         marker='o')

plt.xlabel("Epoch")
plt.ylabel("mAP50")
plt.title("YOLOv8 Training Performance")
plt.grid(True)
plt.show()

metrics = model.val()

print(metrics)

from google.colab import drive
drive.mount('/content/drive')

video_path = "/content/drive/MyDrive/ML_final_data.mp4"

model = YOLO("runs/detect/train/weights/best.pt")

video_path = "/content/drive/MyDrive/ML_final_data.mp4"

import moviepy.editor as mp

video = mp.VideoFileClip(video_path)
short_video = video.subclip(0, 60)  # 前60秒
short_video.write_videofile("test_60s.mp4")

model.predict(
    source="test_60s.mp4",
    save=True,
    conf=0.05,
    vid_stride=5
)

!ls runs/detect

!ls runs/detect/predict

!find runs -type f
