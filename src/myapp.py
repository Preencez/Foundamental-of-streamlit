import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder

# Set the page title and configuration
st.set_page_config(page_title="Item Price Prediction App")

# Load the trained model
model = joblib.load("D:\Projects\LP 2-Linear Regression Project\gbm_model.pkl")

# Define the input and output interfaces for the Streamlit app
st.title("Item Price Prediction App")

# Add a header and description
st.header("Welcome!")
st.write("This app predicts the price of items based on user input.")

# Input fields
st.sidebar.title("Input Features")
city_options = ["Ambato", "Babahoyo", "Cayambe", "Cuenca", "Daule", "El Carmen", "Esmeraldas", "Guaranda",
                "Guayaquil", "Ibarra", "Latacunga", "Libertad", "Loja", "Machala", "Manta", "Playas", "Puyo",
                "Quevedo", "Quito", "Riobamba", "Salinas", "Santo Domingo", "unknown"]
city = st.sidebar.selectbox("City", city_options)
dcoilwtico = st.sidebar.number_input("Crude Oil Price")
family_options = ["AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS", "BREAD/BAKERY", "CELEBRATION",
                  "CLEANING", "DAIRY", "DELI", "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
                  "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES", "HOME CARE", "LADIESWEAR",
                  "LAWN AND GARDEN", "LINGERIE", "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
                  "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY", "PREPARED FOODS", "PRODUCE",
                  "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD"]
family = st.sidebar.selectbox("Family", family_options)
onpromotion = st.sidebar.selectbox("On Promotion", [True, False])
sales = st.sidebar.number_input("Sales")
store_nbr = st.sidebar.number_input("Store Number")
transactions = st.sidebar.number_input("Transactions", min_value=1, max_value=10)
transferred = st.sidebar.selectbox("Transferred", [True, False])
holiday_type_options = ["Normal", "Event", "Holiday", "", "Transfer"]
holiday_type = st.sidebar.selectbox("Holiday Type", holiday_type_options)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100)
month = st.sidebar.number_input('Month', min_value=1, max_value=12)

# Create a DataFrame with the user input
input_data = {
    "city": [city],
    "dcoilwtico": [dcoilwtico],
    "family": [family],
    "onpromotion": [onpromotion],
    "sales": [sales],
    "store_nbr": [store_nbr],
    "transactions": [transactions],
    "transferred": [transferred],
    "holiday_type": [holiday_type],
    "year": [year],
    "month": [month],
}
input_df = pd.DataFrame(input_data)

# Encode the categorical features in the input data
encoder = OrdinalEncoder()
encoder.fit(input_df[['city', 'family', 'onpromotion', 'transferred', 'holiday_type']])

# Prepare the input data for prediction
encoded_features = encoder.transform(input_df[['city', 'family', 'onpromotion', 'transferred', 'holiday_type']])
input_df[['city', 'family', 'onpromotion', 'transferred', 'holiday_type']] = encoded_features

# Make the prediction
prediction = model.predict(input_df)

# Display the prediction
st.subheader("Prediction")
st.write("The predicted price of the item is:", prediction[0])

# Return the user input as a DataFrame
st.subheader("User Input")
st.write("The user input is:")
st.write(input_df)
