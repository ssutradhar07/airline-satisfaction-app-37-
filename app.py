import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ১. মডেল এবং কলাম লিস্ট লোড করা
model = pickle.load(open('airline_model.pkl', 'rb'))
cols = pickle.load(open('columns_list.pkl', 'rb'))

st.set_page_config(page_title="Airline Satisfaction AI", layout="wide")
st.title("✈️ Airline Passenger Satisfaction Predictor")

st.sidebar.header("Input Passenger Details")

# ২. ডাইনামিক ইউজার ইনপুট (সবগুলো ফিচারের জন্য)
input_data = {}
for col in cols:
    if col in ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
        input_data[col] = st.sidebar.number_input(f"Enter {col}", value=0)
    elif col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        # আমরা কোলাবে যেভাবে সংখ্যা দিয়েছিলাম, এখানেও সেভাবে ম্যাপ করছি
        if col == 'Gender': 
            val = st.sidebar.selectbox("Gender", ["Male", "Female"])
            input_data[col] = 0 if val == "Male" else 1
        elif col == 'Customer Type':
            val = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
            input_data[col] = 1 if val == "Loyal Customer" else 0
        elif col == 'Type of Travel':
            val = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
            input_data[col] = 1 if val == "Business travel" else 0
        elif col == 'Class':
            val = st.sidebar.selectbox("Class", ["Business", "Eco Plus", "Eco"])
            input_data[col] = 2 if val == "Business" else (1 if val == "Eco Plus" else 0)
    else:
        input_data[col] = st.sidebar.slider(f"{col} (0-5)", 0, 5, 3)

# ৩. প্রেডিকশন বাটন
if st.button("Predict Satisfaction"):
    # ইনপুট ডেটাকে DataFrame-এ রূপান্তর করা
    df_input = pd.DataFrame([input_data])
    
    # কলামের ক্রম ঠিক রাখা (Model যেভাবে শিখেছে)
    df_input = df_input[cols]
    
    prediction = model.predict(df_input)
    
    if prediction[0] == 1:
        st.success("🎉 The passenger is likely to be SATISFIED!")
    else:
        st.error("😞 The passenger is likely to be NEUTRAL or DISSATISFIED.")