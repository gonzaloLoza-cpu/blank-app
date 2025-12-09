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
categorical_features = ['SEX', 'CURSMOKE', 'PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'DIABETES', 'BPMEDS', 'DEATH', 'ANGINA', 'HOSPMI', 'MI_FCHD', 'ANYCHD', 'CVD', 'HYPERTEN', 'educ' ]
numerical_features = ['AGE', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'TOTCHOL']
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=period_1_relev[feature])
    plt.title(f'Box plot of {feature}')
    st.pyplot(plt)
    plt.clf()
st.write("We tried several methods such as 2*IQR and Z-score but ultimately decided to use biological limits defined by us to cap our data. We used winsorization to impute these values with NaN for now and later re-impute them.")

x_train_processed = x_train.copy(deep=True)
x_test_processed = x_test.copy(deep=True)

winsorization_caps = {
    'TOTCHOL': 500,
    'DIABP': 125,
    'SYSBP': 220,
    'BMI': 50,
    'HEARTRTE': 200
}

for col, cap_val in winsorization_caps.items():
    if col in x_train_processed.columns:
        original_train_count = len(x_train_processed[x_train_processed[col] > cap_val])
        x_train_processed[col] = np.where(x_train_processed[col] > cap_val, np.nan, x_train_processed[col])


    if col in x_test_processed.columns:
        original_test_count = len(x_test_processed[x_test_processed[col] > cap_val])
        x_test_processed[col] = np.where(x_test_processed[col] > cap_val, np.nan, x_test_processed[col])

st.write("After applying winsorization, here is how our graphs look now:")
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=x_train_processed[feature])
    plt.title(f'Box plot of {feature} after Winsorization')
    st.pyplot(plt)
    plt.clf()
st.write("Now that outliers have been handled, we can proceed to impute the missing values.")

st.subheader("Imputing Missing Values")
st.write("Here is a heatmap showing the missing values before imputation:")
plt.figure(figsize=(10,6))
sns.heatmap(x_train_processed.isnull(), cbar = False)
plt.title('Heatmap of missing values')
plt.show()
st.pyplot(plt)
plt.clf()
st.write("We used the most common imputation method to fill in the missing values for our categorical variables and KNN imputation for our numerical variables. When imputing we also made sure to:")
st.markdown("""
- Choose a appropriate method for each type of data
- Avoid data leakage by fitting imputers only on training data and then applying to test data
""")
missing_columns = [col for col in x_train_processed.columns if x_train_processed[col].isnull().any()]
def intersection(lst1, lst2):
  lst3 = [value for value in lst1 if value in lst2]
  return lst3
cat_miss = intersection(missing_columns, categorical_features)
num_miss = intersection(missing_columns, numerical_features)
from sklearn.impute import SimpleImputer, KNNImputer

x_train_final = x_train_processed.copy()
x_test_final = x_test_processed.copy()

# Categorical imputation
if len(cat_miss) > 0:
    cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    x_train_final[cat_miss] = cat_imputer.fit_transform(x_train_final[cat_miss])
    x_test_final[cat_miss] = cat_imputer.transform(x_test_final[cat_miss])

# Numerical imputation
knn_imputer = KNNImputer(n_neighbors=10, weights="uniform")

# Ensure num_miss only contains columns that are actually in x_train_final
num_miss_train = [col for col in num_miss if col in x_train_final.columns]
num_miss_test = [col for col in num_miss if col in x_test_final.columns]

if len(num_miss_train) > 0:
    x_train_final[num_miss_train] = knn_imputer.fit_transform(x_train_final[num_miss_train])

if len(num_miss_test) > 0:
    x_test_final[num_miss_test] = knn_imputer.transform(x_test_final[num_miss_test])
st.write("After imputation, we double checked for any remaining missing values:")
plt.figure(figsize=(10,6))
sns.heatmap(x_train_final.isnull(), cbar = False)
plt.title('Heatmap of missing values')
plt.show()
st.pyplot(plt)
plt.clf()
st.write("No missing values remain. The dataset is now clean and ready for modeling.")


