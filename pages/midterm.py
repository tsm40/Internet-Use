import streamlit as st
import pandas as pd

st.set_page_config(page_title = "Physical Activity's Influence on Internet Addiction - Group 20 Midterm Report")


# TITLE
st.title("Physical Activity's Influence on Internet Addiction - Group 20 Midterm Report")



st.header('Data Preprocessing')
st.subheader('Manual Feature Selection/Hand-Engineered Features')
st.write("The dataset contains many repetitive columns, such as weight, height, and BMI. By manually deleting redundant features (like height and weight due to having a BMI column), we created a more streamlined and useful dataset.")
st.subheader('Filling Missing Data')
st.write("Since our dataset had a lot of missing values, we needed to fill them in strategically. First, we removed columns with more than 50% missing values. Next, we filled in missing data with average values, but to improve precision, we experimented with other imputation methods, like linear regression. However, linear regression can sometimes produce values outside the expected range. We then adjusted our approach by selecting more suitable predictor variables or exploring alternative imputation methods.")
st.subheader("PCA for Dimensionality Reduction")
st.write("PCA was used to reduce the dimensionality of our original dataset. By setting an appropriate threshold of 90% accuracy, we retained 21 principal components (PCs). This significantly reduced the features from the original 51, thus resulting in a more manageable dataset.")

st.header('ML Algorithms/Models Implemented')
st.subheader('Supervised Learning: Logistic regression')
st.write('In our data set, the goal is to categorize each person into ranges of SII values, which is a summary of the total scores each participant received indicating their believed problematic internet usage. In particular, there are 4 score ranges (0 for None, 1 for Mild, 2 for Moderate, and 3 for Severe). Because this is directly translated into a classification problem, we believed logistic regression would be a quick and simple starting point we could use. Moreover, for any other classification model, this simple logistic regression could be a strong benchmark to compare against.')
st.subheader('Unsupervised Learning: Fuzzy K-means Clustering')

st.header('Results and Discussion')