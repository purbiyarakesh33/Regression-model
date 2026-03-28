
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("House Price Predictor")
st.write("Enter house details to predict the sale price")

overall_qual    = st.slider("Overall Quality (1-10)", 1, 10, 5)
year_built      = st.number_input("Year Built", 1900, 2023, 2000)
year_remod      = st.number_input("Year Remodelled", 1900, 2023, 2000)
mas_vnr_area    = st.number_input("Masonry Veneer Area", 0, 1500, 0)
total_bsmt_sf   = st.number_input("Total Basement SF", 0, 3000, 1000)
first_flr_sf    = st.number_input("1st Floor SF", 500, 5000, 1200)
gr_liv_area     = st.number_input("Living Area (sqft)", 500, 6000, 1500)
full_bath       = st.slider("Full Bathrooms", 0, 4, 2)
garage_cars     = st.slider("Garage Cars", 0, 4, 2)
garage_area     = st.number_input("Garage Area (sqft)", 0, 1500, 500)
exter_qual_ta   = st.selectbox("Exterior Quality Average?", [0, 1])
foundation_pconc = st.selectbox("Concrete Foundation?", [0, 1])
bsmt_qual_ex    = st.selectbox("Basement Quality Excellent?", [0, 1])
kitchen_qual_ex = st.selectbox("Kitchen Quality Excellent?", [0, 1])
kitchen_qual_ta = st.selectbox("Kitchen Quality Average?", [0, 1])

if st.button("Predict Price"):
    features = np.array([[
        overall_qual, year_built, year_remod, mas_vnr_area,
        total_bsmt_sf, first_flr_sf, gr_liv_area, full_bath,
        garage_cars, garage_area, exter_qual_ta, foundation_pconc,
        bsmt_qual_ex, kitchen_qual_ex, kitchen_qual_ta
    ]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    st.success(f"Predicted House Price: ${prediction[0]:,.0f}")