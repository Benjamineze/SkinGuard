import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
import base64  # Import the base64 module


# Load the pre-trained model
with open('lr_mod_ada_skinca_.pkl', 'rb') as file:
    data = pickle.load(file)
    lr_mod_adasyn = data['model']

#  custom CSS to add the background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        data = file.read()
        encoded_image = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                image-rendering: optimizeQuality; /* Ensures the best quality rendering */
            }}
            </style>
            """,
            unsafe_allow_html=True
        )


# Call the function to add the background image
add_bg_from_local("images (3).jfif")  # Ensure the path is correct


def add_circle_image_to_sidebar(image_path):
    # Read the image and encode it to base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Add the image with custom CSS to style it as a circle
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: flex-start; align-items: center;">
            <img src="data:image/png;base64,{encoded_image}" 
                 style="border-radius: 50%; width: 100px; height: 100px; object-fit: cover; margin-bottom: 20px;">
        </div>
        """,
        unsafe_allow_html=True
    )

# Call the function with the path to the image
add_circle_image_to_sidebar("images (2).jfif")


# Define the user input function
def get_user_input():
    st.sidebar.header('Enter User Data')

    checkup = st.sidebar.selectbox('Checkup', ['Never', '5 or more years ago', 'Within the past 5 years', 
                                               'Within the past 2 years', 'Within the past year'])
    exercise = st.sidebar.selectbox('Exercise', ['No', 'Yes'])
    heart_disease = st.sidebar.selectbox('Heart Disease', ['No', 'Yes'])
    other_cancer = st.sidebar.selectbox('Other Cancer', ['No', 'Yes'])
    depression = st.sidebar.selectbox('Depression', ['No', 'Yes'])
    diabetes = st.sidebar.selectbox('Diabetes', ['No', 'Yes'])
    arthritis = st.sidebar.selectbox('Arthritis', ['No', 'Yes'])
    age_category = st.sidebar.selectbox('Age Category', ['18-24', '25-29', '30-34', '35-39', '40-44', 
                                                         '45-49', '50-54', '55-59', '60-64', '65-69', 
                                                         '70-74', '75-79', '80+'])

    # Create a dictionary with the input data
    user_data = {'Checkup': checkup, 'Exercise': exercise, 'Heart_Disease': heart_disease,
                 'Other_Cancer': other_cancer, 'Depression': depression, 'Diabetes': diabetes, 
                 'Arthritis': arthritis, 'Age_Category': age_category}

    # Convert to DataFrame
    features = pd.DataFrame([user_data])
    return features

# Process user input for prediction
def process_input(user_df):
    # Mapping categorical inputs to numerical
    size_map_checkup = { 'Never': 0, '5 or more years ago': 1, 'Within the past 5 years': 2,
                        'Within the past 2 years': 3, 'Within the past year':4 }

    size_map_age = { '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44':4, '45-49': 5, 
                    '50-54': 6, '55-59':7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80+': 12}

    user_df['Checkup'] = user_df['Checkup'].map(size_map_checkup)
    user_df['Age_Category'] = user_df['Age_Category'].map(size_map_age)

    cat_cols = user_df.select_dtypes(include=['object']).columns
    user_df[cat_cols] = user_df[cat_cols].apply(lambda x: x.replace({'Yes': 1, 'No': 0}))

    return user_df

# Main function
def main():
    st.title("Skin Cancer Prediction App")

    # Get user input
    user_df = get_user_input()

    # Display user input
    st.subheader('User Input:')
    st.write(user_df)

    # Process input for prediction
    processed_input = process_input(user_df)

    # Make predictions and obtain probabilities
    Skin_Cancer_proba = lr_mod_adasyn.predict_proba(processed_input)[:, 1]  

    # Make a decision based on a threshold 
    threshold = 0.5
    Skin_cancer = 1 if Skin_Cancer_proba > threshold else 0

    # Output results
    st.subheader('Result:')
    if Skin_cancer == 1:
       st.markdown(
        f'<p style="font-size:24px; font-weight:bold; font-style:italic; color:red">'
        f'This patient is diagnosed with Skin cancer with a Probability of {Skin_Cancer_proba[0]*100:.2f}%</p>',
        unsafe_allow_html=True
    )
    else:
        st.markdown(
        f'<p style="font-size:24px; font-weight:bold; font-style:italic; color:green">'
        f'This patient does not have Skin cancer with a Probability of {(1 - Skin_Cancer_proba[0])*100:.2f}%</p>',
        unsafe_allow_html=True
    )
if __name__ == '__main__':
    main()

