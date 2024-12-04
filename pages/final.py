import streamlit as st
import pandas as pd

st.set_page_config(page_title = "Physical Activity's Influence on Internet Addiction - Group 20 Final Report")


# TITLE
st.title("Physical Activity's Influence on Internet Addiction - Group 20 Final Report")

url1 = "https://github.com/tsm40/Internet-Use"
st.write("[GitHub link](%s)"%url1)

st.header('Data Preprocessing')

st.subheader('Data Exploration')
st.write("We first examined the data we were working with to get a general sense of what features should be used for training and what preprocessing needed to be done. We noticed that the target variable SII value has a very uneven distribution, which we must keep in mind for the interpretability of our models. Other things of note were that our dataset contained many null values, which necessitates imputation. Finally, there was a significant portion of the data containing the exact values of the answers to the PCIAT questionnaire, which directly contributes to the mathematical calculation of the SII values. Thus, we decided to manually remove these columns so that our models will not be influenced by those direct values and instead will be trained more intelligently on more biometric data.")
st.image("Figures/distribution.png", caption="SII value category distribution in training dataset.")
st.subheader('Manual Feature Selection/Hand-Engineered Features')
st.write("The dataset contains many repetitive columns, such as weight, height, and BMI. By manually deleting redundant features (like height and weight due to having a BMI column), we created a more streamlined and useful dataset.")
st.subheader('Filling Missing Data')
st.write("Since our dataset had a lot of missing values, we needed to fill them in strategically. First, we removed columns with more than 50% missing values. Next, we filled in missing data with average values, but to improve precision, we experimented with other imputation methods, like linear regression. However, linear regression can sometimes produce values outside the expected range. We then adjusted our approach by selecting more suitable predictor variables or exploring alternative imputation methods.")
st.subheader("PCA for Dimensionality Reduction")
st.write("PCA was used to reduce the dimensionality of our original dataset. By setting an appropriate threshold of 90% accuracy, we retained 21 principal components (PCs) composed of 42 features. This significantly reduced the number of features to focus on, thus resulting in a more manageable dataset.")
col1, col2 = st.columns(2)
with col1:
    st.image("Figures/CEV_PCA.png")
with col2:
    st.image("Figures/EV_PCA.png")

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
    st.write("Accuracy: 73%")
    st.write("Precision: macro: 51%, weighted: 70%")
    st.write("Recall: macro: 33%, weighted: 73%")
    st.subheader("Model Analysis")
    st.write("For logistic regression, while the accuracy was decent for the 4 categories, the recall and precision were not. This was mainly because the model overly preferred choosing category 0. As for why this happened, looking deeper into the data, we see that of the actual labels, 371 of the 554 belonged to range 0. This imbalance of data could explain the issues in precision and recall. As for choosing the features themselves, we chose the features from the principal components that had the highest absolute loading values and thus the most impact on the principal components. Specifically, these features were, SDS-SDS_Total_T, BIA-BIA_FFMI, Physical-Diastolic_BP, and PreInt_EduHx-computerinternet_hoursday. Note, while PAQ_A-PAQ_A_Total had one of the top 5 absolute loading values, we did not use this value as it also had many null values in the dataset.")
    st.markdown("*(change tabs above for other model)*")
with fuzzy_tab:
    st.subheader("Model Results")
    st.image("Figures/fuzzy_conf.png", caption="Fuzzy K-means clustering confusion matrix.")
    st.write("Accuracy: 39.7%")
    st.write("Precision: macro: 23.7%, weighted: 42.2%")
    st.write("Recall: macro: 21%, weighted: 39.7%")
    st.subheader("Model Analysis")
    st.write("For Fuzzy K-means, it is quite noticeable that the accuracy, precision, and recall metrics are all quite low, especially compared to logistic regression. We dropped all samples that had NaN values for the PCIAT-PCIAT_Total column, as this column helps us obtain our model accuracy as it represents the true cluster value (0-3) for each sample. We chose to use the 'BIA-BIA_ICW', 'BIA-BIA_TBW', 'BIA-BIA_FFM',  'BIA-BIA_BMR', and 'BIA-BIA_LST' features as these 5 features contributed the most to the first iteration of PCA. We noticed that adding additional features to our Fuzzy K-means model seemed to significantly decrease the training accuracy, and most of the samples we did predict the clustering for had NaN values for at least one of the 5 features. Imputation for the missing data in these samples may significantly improve the training and testing accuracies. We also noticed that the dataset resulting from the 2736 individuals who did have the PCIAT-PCIAT_Total was quite unbalanced as more than 50% of the samples had a true cluster value of 0 and only 34 individuals had a true cluster value of 3.")
    st.markdown("*(change tabs above for other model)*")

st.header("Next Steps")
st.subheader("Data Imputation")
st.write("The next step in imputation is to identify appropriate predictors to improve the precision of missing value predictions by linear regression. We can also utilize KNN (K-Nearest Neighbors) to fill in the missing values. Additionally, we can apply different imputation methods to specific features as needed.")
st.subheader("Logistic Regression")
st.write("The next step for improving logistic regression would be adding balance to the dataset. We believe we can take two steps to do this. Firstly, we could subtract data points that belong to the first range of SII values. Secondly, with value imputation, we could gain a few data points back, which were previously removed due to there being null values. When added back, we would be able to gain data points that belong to other ranges.")
st.subheader("Fuzzy K-Means Clustering")
st.write("Given our lower-than-expected performance for our fuzzy K-means clustering model, we could take several steps to improve them by utilizing our PCA and other preprocessing techniques more effectively and handling missing values more rigorously. In terms of preprocessing, we could select our features more carefully by better incorporating PCA outputs with our K-means inputs rather than just picking the top few features. When it comes to missing values, experimenting with more advanced implementations like iterative imputation or K-nearest neighbors could provide a better representation of the underlying data. Utilizing supervised learning may be very helpful for Fuzzy K-means clustering.")
st.subheader("Support Vector Machine")
st.write("We will implement a support vector machine-based model to predict the SII value. This model will be evaluated using cross-validation accuracy and will be compared to the models we have implemented thus far.")

st.link_button("Gaant Chart", "https://docs.google.com/spreadsheets/d/1NzZaNOShu23aP_18EbaMfo7mWibpnGhX/edit?usp=sharing&ouid=111674034097172792134&rtpof=true&sd=true")
df = pd.DataFrame([['Chen Ye', 'Value imputation exploration and implementation'],
                   ['Matthew Lei', 'Fuzzy K-means exploration and implementation'],
                   ['Eric Ma', 'Logistic regression exploration and implementation'],
                   ['Tiffany Ma', 'PCA feature reduction exploration and implementation, Streamlit'],
                   ['Kevin Song', 'Fuzzy K-means exploration and implementation']],
                  columns = ["Team Member", "Proposal Contributions"])
st.table(df)
