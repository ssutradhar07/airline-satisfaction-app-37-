import streamlit as st
import pandas as pd
import pickle
import os

# --- পেজ কনফিগারেশন ---
st.set_page_config(page_title="Airline Satisfaction AI", layout="centered")

# --- ফাইল লোড করার ফাংশন (নিরাপদ উপায়) ---
def load_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'airline_model.pkl')
    cols_path = os.path.join(current_dir, 'columns_list.pkl')
    
    try:
        model = pickle.load(open(model_path, 'rb'))
        cols = pickle.load(open(cols_path, 'rb'))
        return model, cols
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, cols = load_files()

# --- অ্যাপ টাইটেল ---
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.write("যাত্রীর তথ্য দিন এবং ফলাফল দেখুন:")

if model and cols:
    # --- ইউজার ইনপুট ফর্ম ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        with col1:
            if 'Gender' in cols:
                gender = st.selectbox("Gender", ["Male", "Female"])
                input_data['Gender'] = 0 if gender == "Male" else 1
            
            if 'Customer Type' in cols:
                cust_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
                input_data['Customer Type'] = 1 if cust_type == "Loyal Customer" else 0
                
            if 'Age' in cols:
                input_data['Age'] = st.number_input("Age", 7, 85, 30)
                
            if 'Type of Travel' in cols:
                travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
                input_data['Type of Travel'] = 1 if travel_type == "Business travel" else 0

        with col2:
            if 'Class' in cols:
                flight_class = st.selectbox("Class", ["Business", "Eco Plus", "Eco"])
                input_data['Class'] = 2 if flight_class == "Business" else (1 if flight_class == "Eco Plus" else 0)
            
            if 'Flight Distance' in cols:
                input_data['Flight Distance'] = st.number_input("Flight Distance", 100, 5000, 1000)
            
            if 'Departure Delay in Minutes' in cols:
                input_data['Departure Delay in Minutes'] = st.number_input("Departure Delay", 0, 1500, 0)
            
            if 'Arrival Delay in Minutes' in cols:
                input_data['Arrival Delay in Minutes'] = st.number_input("Arrival Delay", 0, 1500, 0)

        st.markdown("---")
        st.write("**সেবার মান রেটিং দিন (0-5):**")
        
        # বাকি রেটিং ফিচারগুলো স্লাইডার হিসেবে আসবে
        for col in cols:
            if col not in input_data:
                input_data[col] = st.slider(f"{col}", 0, 5, 3)

        # প্রেডিকশন বাটন
        submit = st.form_submit_button("Predict Result")

    # --- ফলাফল প্রদর্শন ---
    if submit:
        # ডেটাফ্রেম তৈরি এবং কলামের সিরিয়াল ঠিক করা
        df_input = pd.DataFrame([input_data])[cols]
        
        prediction = model.predict(df_input)
        
        if prediction[0] == 1:
            st.success("🎉 যাত্রী সন্তুষ্ট (SATISFIED)!")
        else:
            st.error("😞 যাত্রী সন্তুষ্ট নন (NEUTRAL or DISSATISFIED)!")

else:
    st.warning("মডেল ফাইল লোড করা সম্ভব হয়নি। অনুগ্রহ করে আপনার GitHub রিপোজিটরি চেক করুন।")
