import streamlit as st
import pandas as pd

# ------ PART 1 ------


# TITLE
st.title('Problematic Internet Use - Group 20 Project Proposal')



st.header('Introduction')
st.subheader('Literature Review')
url1 = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7330165/"
st.write('[This paper](%s) explores the impacts of physical fitness on reducing internet addiction. The paper finds that physical activity can cause key changes in brain chemistry concerning internet addiction. Specifically, they find that fitness can improve things like attention span and dopamine production, which can be correlated with addiction.' % url1)
url2 = "https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-024-18474-1"
st.write('[This paper](%s) argues that there is a strong case for the negative relationship between physical activity and internet addiction. Specifically, they find that p<0.01 for the relationship between physical activity and internet addiction.' % url2)

st.subheader('Dataset Description')
st.write('The Dataset we chose was the Child Mind Institute - Problematic Internet Use data set. The data set provides physical activity markers as well as internet usage markers. The fitness markers are based on fitness gram measurements, sleep disturbance, and bioelectric markers. The dataset measures internet usage through total internet usage and an internet addiction scale called the Severity Impairment Index (`sii`).')

# * optional kwarg unsafe_allow_html = True

# Interactive widgets
st.link_button('Dataset Link', "https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data")

st.header('Problem Definition')
st.write('From our literature review, it is clear that internet addiction and physical health are negatively correlated. From here, we want to explore how much these specific physical attributes factor into predicting internet addiction by training models to predict `sii`. For example, we could explore how accurately sleep disturbance may predict internet addiction as opposed to general physical activity. To achieve this, we plan to apply various data preprocessing methods on the dataset, followed by running logistic regression, an SVM, and K-means clustering.')
st.write('As students, we often don’t have time to exercise daily. Additionally, many of us tend not to focus on the impacts that our lifestyle choices could have on us. As such, we wanted to explore how a sedentary lifestyle could trickle into other issues, namely how much time we spend online. Even beyond college students, this type of study can give crucial information on the environment children are raised in today.')

st.header('Methods')
st.subheader('Data Preprocessing Methods')
st.write('*Standardization/Normalization:* Ensures that the features used are on a similar scale no matter the algorithm, as physical activity and internet usage data can have different value ranges. Thus, it alleviates the shortcomings of algorithms sensitive to small changes in inputs.')
st.write('*Principal Component Analysis (PCA):* Reduces the dimensionality of our data while retaining essential features, which can help visualize the data better and use it more effectively.')
st.write('*Missing Value Imputation:* Replaces some of the missing values in the dataset with estimated values to ensure efficient model training without losing valuable information.')\

st.subheader('ML Algorithms and Models')
st.write('*Logistic Regression:* Use for classifying distinct categories of impairment')
st.write('*Support Vector Machine (SVM):* Multi-class classifier that can handle the complex, possibly non-linear relationship of physical activity indicators and problematic internet use in children')
st.write('*Fuzzy K-Means Clustering:* Assigns a probability of a point belonging to each cluster, thus accounting for data points being in two or more clusters or having multiple `sii` values')

st.header('Results and Discussion')
st.subheader('Quantitative Metrics')
st.write('*Mean Squared Error (MSE):* Finding the average squared difference between expected and actual impairment can show how right or wrong our expectations are.')
st.write('*Cross-Validation Accuracy:* Compare the efficiency of our different models used and produce less biased models')
st.write('*Silhouette Coefficient:* Evaluate the clustering quality of Fuzzy K-Means to show how similar objects are in their clusters compared to others')

st.subheader('Project Goals and Expected Results')
st.write('We hope to achieve an accuracy score of 0.7 or higher for these metrics ability to predict `sii` but anticipate that attributes like physical activity levels and bioelectric signals can overlap in terms of impact on internet use, leading to misclassification. Thus, we expect subgroups with different disturbance or stress levels to be difficult to differentiate. Therefore, we will use confusion matrices to identify patterns where the model struggles most.')
