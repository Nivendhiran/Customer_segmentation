import streamlit as st
import pandas as pd
import numpy as np
import joblib


scalar=joblib.load("scalar.pkl")
kmeans=joblib.load("kmeans_model.pkl")

cluster_labels={
    0:"Budget Customer",
    1:"Standard Customer",
    2:"Target Customer(High Income & spending score is high)",
    3:"Potential Customer(High Income & Low spending)",
    4:"Low Income & High spending"
}
#stramlit UI

st.title("Customer Segmentation Using K-Means")
st.markdown("Enter new Customer Details to Predict their Segment")

#user input

income=st.number_input("Annual Income (k$)", min_value=10, max_value=150, value=50)
spending=st.number_input("Spending Score (1-100)",min_value=1, max_value=100, value=50)

#predict cluster
if st.button("predict cluster"):
    new_data=pd.DataFrame([[income,spending]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    new_scaled=scalar.transform(new_data)
    cluster=kmeans.predict(new_scaled)[0]
    st.success(f"predcit cluster:{cluster}-{cluster_labels.get(cluster,'unknown')}")