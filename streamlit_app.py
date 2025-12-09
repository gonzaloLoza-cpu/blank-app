import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

st.title("🎈 My new streamlit app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.title("Project Stroke Analysis of Smokers")
st.write(
    "This app analyzes risk of stroke based on data from the Framingham Heart Study."
)


dataset = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')
st.subheader("Dataset Overview")
st.write(dataset.head())