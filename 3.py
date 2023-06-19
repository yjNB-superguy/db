import streamlit as st
import os
from fastai.vision.all import *
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
from PIL import Image, ImageEnhance
import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16

import pathlib
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

model = VGG16(weights="imagenet")
st.set_page_config(page_title="图像转换编辑器", page_icon=":eyeglasses:")
st.title("图像分类转换器")


# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, "export.pkl")

# Load the model
learn_inf = load_learner(model_path)

st.title("Image Classification App")
st.write("Upload an image and the app will predict the corresponding label.")

# Allow the user to upload an image
uploaded_file = st.file_uploader("上传一个图像", type=["png", "jpg", "jpeg"])
if not uploaded_file:
    st.warning("请上传一张图像。")
    st.stop()
# If the user has uploaded an image
if uploaded_file is not None:
    # Display the image
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get the predicted label
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")

    # 对上传的图像进行处理和显示
    original_image = Image.open(uploaded_file)
    st.image(original_image, use_column_width=True, caption="原始图像")

    # 对图像进行编辑和处理，省略了调整亮度和对比度等功能
    filtered_image = original_image.filter(ImageFilter.FIND_EDGES)
    cropped_image = filtered_image.crop((100, 100, 400, 400))
    resized_image = cropped_image.resize((224, 224), Image.BICUBIC)
    color_image = ImageOps.posterize(resized_image, 4)

    # 显示最终结果
    st.image(color_image, use_column_width=True, caption="最终结果")
