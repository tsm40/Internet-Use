#smote and scale
import pandas as pd
from NeuralNetFuncs import preprocess_data
from imblearn.over_sampling import SMOTE
train = pd.read_csv("NeuralNet/filled_data.csv")
X_train, y_train = preprocess_data(train)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


X_train_df = pd.DataFrame(X_train_resampled)
y_train_df = pd.DataFrame(y_train_resampled, columns=['sii']) 
combined_df = pd.concat([X_train_df, y_train_df], axis=1)

combined_df.to_csv('NeuralNet/train_after_smote.csv', index=False)

test = pd.read_csv("NeuralNet/filled_test_data.csv")
X_test, y_test = preprocess_data(test)
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test, columns=['sii']) 
combined_df2 = pd.concat([X_test_df, y_test_df], axis=1)
combined_df2.to_csv('NeuralNet/test_after_preprocess.csv', index=False)
