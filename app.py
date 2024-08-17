import pickle
import pandas as pd
import numpy as np
import sklearn
import imblearn
import streamlit as st
from PIL import Image

# Title of the Web App
st.title("Credit Card Fraud Detection Model")

# Display an image
image = Image.open("images.jpg")
st.image(image, caption='Fraud Detection', use_column_width=True)

# Get user input
input_df = st.text_input("Please provide all the required feature details separated by commas:")

submit = st.button("Submit")

if submit:
    try:
        # Load the model
        model = pickle.load(open('resampled_rf_model.pkl', 'rb'))

        # Convert input string to a numpy array
        features = np.asarray(input_df.split(','), dtype=np.float64)
        
        # Ensure the correct number of features
        if features.shape[0] != model.n_features_in_:
            st.error(f"Expected {model.n_features_in_} features, but got {features.shape[0]}.")
        else:
            # Reshape and predict
            prediction = model.predict(features.reshape(1, -1))

            # Display the result
            if prediction[0] == 0:
                st.success("Legitimate Transaction")
            else:
                st.error("Fraudulent Transaction")
                
    except ValueError:
        st.error("Please ensure all feature details are provided in numeric format.")
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")