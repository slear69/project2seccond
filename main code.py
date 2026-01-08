import streamlit as st
import numpy as np
from PIL import Image
import joblib
import requests
import io
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="")
st.title(" Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")
# Simple model loading with fallback
@st.cache_resource
def load_model():
try:
#Try to load pre-trained model
#Using sklearn's built-in digits dataset
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
digits load_digits()
X digits.images.reshape((len(digits.images), -1)) / 16.0
y = digits.target
X_train, _y_train, = train_test_split(X, y, test_size=0.2, random_state=42)
