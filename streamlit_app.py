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
st.write("We had 3 Periods worth of data due to overlap and bias when using all three we decided to use Period 1 only for our analysis.")
st.markdown("""
- Most active participants; 4,434 entries
- Highest incidence of stroke; 5.5%
- Least releavant missing data; only 4 relevant features with missing values
""")
period_1 = dataset.groupby('PERIOD').get_group(1)


st.subheader("Deciding on Features to Use")
st.write("We decided to drop the following features due to irrelevance or just inaccuracy for stroke prediction:")
period_1_relev = period_1.drop(columns=["HDLC", "LDLC", "RANDID", "TIME",
    "TIMESTRK",
    "TIMEAP",
    "TIMEMI",
    "TIMEMIFC",
    "TIMECHD",
    "TIMESTRK",
    "TIMECVD",
    "TIMEDTH",
    "TIMEHYP",
    "GLUCOSE"])
st.write("HDLC", "LDLC", "RANDID", "TIME",
    "TIMESTRK",
    "TIMEAP",
    "TIMEMI",
    "TIMEMIFC",
    "TIMECHD",
    "TIMESTRK",
    "TIMECVD",
    "TIMEDTH",
    "TIMEHYP",
    "GLUCOSE")
st.write("The resulting dataset is as follows:")
st.write(period_1_relev.head())
st.subheader("Train Test Split")
st.write("We performed an 80-20 train-test split on the dataset to prepare for modeling.")
from sklearn.model_selection import train_test_split
import pandas as pd
y = period_1_relev.STROKE  # Labels
x = period_1_relev.drop(columns=["STROKE"])  # Features
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
st.subheader("Handling Missing Data")
st.write("We noticed that there were some missing values in the dataset. Here is a summary of the missing values:")
missing_summary = period_1_relev.isnull().sum()
st.write(missing_summary[missing_summary > 0])
st.write("These would have to be imputed however before we could impute, we have to remove outliers as to not skew our imputation.")
st.subheader("Outlier Detection and Removal")
st.write("We used box plots to identify outliers in the numerical features.")
numerical_features = ['AGE', 'TOTCHOL', 'BMI', 'SYSBP', 'DIABP', 'CIGPDAY']
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=period_1_relev[feature])
    plt.title(f'Box plot of {feature}')
    st.pyplot(plt)
    plt.clf()
st.write("Based on the box plots, we defined outliers as values that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR. We removed these outliers from the dataset.")
def remove_outliers(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df