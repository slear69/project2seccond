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
        X_train, _, y_train, = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MLPClassifier(hidden_layer_sizes=(100,),max_iter=100,random state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Model loading error: (e)")
        return None
model = load_model()

if model is None:
  st.warning("nah twin no model :)")
else:
  st.success("twin we made it :D")

uploaded_file=st.file_uploader("twin now gime a image file", type=["png","jpg","jpeg"])
if uploaded file is not None:
#Display the uploaded image
    Image Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image', use_column_width=True)
    try:
        # Convert to grayscale and resize to 8x8
      img gray image.convert('L')
      img_resized img gray.resize((8, 8))
      #Convert to numpy array and invert if needed
      np.array(img_resized) img_array
      # If background is dark, invert
      if np.mean(img_array) > 128:
         img_array 255 - img_array
      #Normalize like the training data
      img_array img_array / 16.0
      img_flat img_array.flatten().reshape(1, -1)
      if model is not None:
          #Make prediction
          prediction model.predict(img_flat) [0]
          st.write(f"## Prediction: **{prediction}**")
          # Show probabilities
          probs model.predict_proba(img_flat) [0]
          st.write("### Probabilities:")
          for i, prob in enumerate (probs):
             st.write(f"Digit (i): (prob:.2%)")
      else:
         #Fallback: simple threshold-based recognition
         st.write("## Using fallback recognition")
         #Simple heuristic based on pixel intensity
         st.write(f"Digit {i}: (prob:.2%}")
      else:
          # Fallback: simple threshold-based recognition.
          st.write("## Using fallback recognition")
          # Simple heuristic based on pixel intensity
          digit_guess = np.argmax(np.sum(img_array.reshape (8, 8), axis=0)) % 10
          st.write(f"Estimated digit: **{digit_guess)**")
    except Exception as e:
        st.error(f"Error processing image: {e}")
# Instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of a handwritten digit (0-9)
2. The image will be resized to 8x8 pixels
3. AI model will predict the digit
4. For best results:
White background
Black digit
Centered digit
Minimal noise""")
