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

st.subheader('Supervised Learning: Logistic Regression')
st.write('In our data set, the goal is to categorize each person into ranges of SII values, which is a summary of the total scores each participant received indicating their believed problematic internet usage. In particular, there are 4 score ranges (0 for None, 1 for Mild, 2 for Moderate, and 3 for Severe). Because this is directly translated into a classification problem, we believed logistic regression would be a quick and simple starting point we could use. Moreover, for any other classification model, this simple logistic regression could be a strong benchmark to compare against.')
st.subheader('Unsupervised Learning: Fuzzy K-means Clustering')
st.write("Since the target variable SII value is categorical, the prediction task is well-suited to a k-means clustering method, where the clusters represent SII value score ranges. To categorize SII values into four score ranges (0-3), we applied fuzzy K-means clustering using k = 4. After preprocessing the data using missing value imputation and feature scaling, we chose the five most relevant features based on the preprocessing (BIA-BIA_ICW, BIA-BIA_TBW, BIA-BIA_FFM, BIA-BIA_BMR, and BIA-BIA_LST). Then, we calculated the membership probabilities for each sample and assigned each data point to the cluster it had the highest probability of being in based on those features. ")

st.header('Results and Discussion')

logreg_tab, fuzzy_tab = st.tabs(["Logistic Regression", "Fuzzy K-means Clustering"])
with logreg_tab:
    st.subheader("Model Results")
    st.image("Figures/logreg_conf.png", caption="Logistic regression confusion matrix.")
    st.write("Accuracy: 39.7%")
    st.write("Precision: macro: 23.7%, weighted: 42.2%")
    st.write("Recall: macro: 21%, weighted: 39.7%")
    st.subheader("Model Analysis")
    st.write("For logistic regression, while the accuracy was decent for the 4 categories, the recall and precision were not. This was mainly because the model overly preferred choosing category 0. As for why this happened, looking deeper into the data, we see that of the actual labels, 371 of the 554 belonged to range 0. This imbalance of data could explain the issues in precision and recall. As for choosing the features themselves, we chose the features from the principal components that had the highest absolute loading values and thus the most impact on the principal components. Specifically, these features were, SDS-SDS_Total_T, BIA-BIA_FFMI, Physical-Diastolic_BP, and PreInt_EduHx-computerinternet_hoursday. Note, while PAQ_A-PAQ_A_Total had one of the top 5 absolute loading values, we did not use this value as it also had many null values in the dataset.")
with fuzzy_tab:
    st.subheader("Model Results")
    st.image("Figures/fuzzy_conf.png", caption="Fuzzy K-means clustering confusion matrix.")
    st.write("Accuracy: 39.7%")
    st.write("Precision: macro: 23.7%, weighted: 42.2%")
    st.write("Recall: macro: 21%, weighted: 39.7%")
    st.subheader("Model Analysis")
    st.write("TBD")

st.header("Next Steps")
st.subheader("Data Imputation")
st.subheader("Logistic Regression")
st.subheader("Fuzzy K-Means Clustering")
st.subheader("Support Vector Machine")

st.link_button("Gaant Chart", "https://docs.google.com/spreadsheets/d/1NzZaNOShu23aP_18EbaMfo7mWibpnGhX/edit?usp=sharing&ouid=111674034097172792134&rtpof=true&sd=true")
df = pd.DataFrame([['Chen Ye', 'Video'], ['Matthew Lei', 'Methods'], ['Eric Ma', 'Dataset'], ['Tiffany Ma', 'GitHub/Streamlit'], ['Kevin Song', 'Introduction/Background']], columns = ["Team Member", "Proposal Contributions"])
st.table(df)