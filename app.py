import streamlit as st 
import pickle
import numpy as np
import pandas as pd 

# Streamlit app
st.title("Car Price Predictor for Selling")

# model 
df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

def create_car_mapping(car_df):
    return car_df.groupby("company")["name"].apply(list).to_dict()


# Load the pickled DataFrame
@st.cache_data
def load_data(pickle_file):
    with open(pickle_file, "rb") as file:
        return pickle.load(file)

# Path to your pickle file
pickle_file = "df.pkl"  # Update with the actual path of your file

# Load the dataset
car_df = load_data(pickle_file)

# Ensure the DataFrame has the necessary columns
if "company" in car_df.columns and "name" in car_df.columns:
    # Create car mapping dynamically
    car_mapping = create_car_mapping(car_df)

    # Select car company
    company = st.selectbox("Select a Car Company", options=list(car_mapping.keys()))

    # Get models for the selected company
    car_models = car_mapping.get(company, [])

    # Select car model
    name = st.selectbox("Select a Car Model", options=car_models)

    # Display selected company and model
    # st.write(f"You selected: {selected_company} - {selected_model}")
else:
    st.error("The dataset does not have the required columns: 'Car_Company' and 'Car_Model'")


year = st.selectbox('Model Year', df['year'].unique())
kms_driven = st.number_input("Kms Driven", min_value=0, step=1, value=0, format="%d")

fuel_type = st.selectbox('Fuel Type', df['fuel_type'].unique())

input_data = pd.DataFrame({'company': [company], 'name': [name], 'year': [year], 'kms_driven': [kms_driven], 'fuel_type': [fuel_type]})


if st.button('Price Predict'):

    prediction = pipe.predict(input_data)
    # Replace negative price with 0
    prediction = max(0, prediction[0])
    st.title(f"Predicted Price: {int(prediction)} INR")

    # Debug
    # print(type(input_data))
    # print(input_data.head())  # Check input structure
