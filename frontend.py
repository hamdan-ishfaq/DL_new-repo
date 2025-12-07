import streamlit as st
import torch
import numpy as np
import tifffile as tiff
from PIL import Image
from model_custom6 import AttentionResUNet

st.set_page_config(page_title="Deforestation Detection")

device = "cuda" if torch.cuda.is_available() else "cpu"

net = AttentionResUNet(in_channels=9, out_channels=2).to(device)
net.load_state_dict(torch.load("model_final.pth", map_location=device))
net.eval()

def load_tif(file):
    arr = tiff.imread(file)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)

def normalize(x):
    x = x / np.max(x)
    return x

st.title("Deforestation Detector - AttentionResUNet")

uploaded = st.file_uploader("Upload 9-band Sentinel TIFF", type=["tif", "tiff"])

if uploaded:
    arr = load_tif(uploaded)
    st.write("Loaded image shape:", arr.shape)

    if arr.shape[0] != 9:
        st.error(f"Model expects 9 channels, got {arr.shape[0]}")
    else:
        arr = normalize(arr)
        x = torch.tensor(arr).unsqueeze(0).to(device)

        with torch.no_grad():
            seg_final, aux_seg, global_pred = net(x)

        mask = seg_final.softmax(dim=1)[0, 1].cpu().numpy()
        mask_img = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_img)

        st.image(mask_img, caption="Deforestation Mask", use_column_width=True)

        st.subheader("Global Prediction")
        st.write(f"Deforested Probability: **{float(global_pred[0]):.4f}**")
