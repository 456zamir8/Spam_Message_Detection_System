import streamlit as st
import pandas as pd 
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model and vectorizer
model = joblib.load("spam_detection_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load your dataset (assumed to be available)
data = pd.read_csv(r"C:\Users\zamir\zamir\data analysis\SMS_Spam\spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Map labels to binary
data['target'] = data['label'].map({'ham': 0, 'spam': 1})
data['message_length'] = data['message'].apply(len)

# Title
st.title("📩 Spam Message Classifier")

# Sidebar
st.sidebar.title("📊 Data Visualizations")
show_charts = st.sidebar.checkbox("Show Visualizations", value=True)

# Prediction input
input_text = st.text_area("✍️ Enter a message to classify:")

if st.button("🔍 Predict"):
    result = vectorizer.transform([input_text])
    prediction = model.predict(result)[0]
    if prediction == 1:
        st.markdown(
            "<div style='background-color:#ffcccc; padding:10px; border-radius:5px;'>"
            "<strong style='color:red;'>🚨SPAM</strong></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#ccffcc; padding:10px; border-radius:5px;'>"
            "<strong style='color:green;'>✅NOT SPAM</strong></div>",
            unsafe_allow_html=True
        )


# Visualizations
if show_charts:
    st.subheader("Spam vs Ham Distribution")
    fig1 = px.pie(data, names='label', title="Spam vs Ham Ratio", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig1)

    st.subheader("Message Length Distribution")
    fig2 = px.histogram(data, x="message_length", color="label",
                        title="Distribution of Message Length by Type",
                        labels={"message_length": "Message Length", "label": "Message Type"},
                        nbins=50, color_discrete_sequence=['blue', 'red'])
    st.plotly_chart(fig2)
