import numpy as np
import seaborn as sns
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [10,15]
#matplotlib inline
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category='FutureWarning')
#importing file
train_data=pd.read_csv ("train.csv")

#TRAIN DATA EDA
print(train_data.head())
train_data=train_data.drop(["Cabin", "PassengerId", "Name", "Ticket"], axis=1)
print(train_data.info())
print("Train_data:" , train_data.shape)

#Checking missing data by heatmap
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False,cmap="tab20c_r")
plt.title("Missing Data of training set")
plt.show()

#Draw boxplot of age to fill according to the gender
plt.figure(figsize=(10,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = train_data). set_title("TRAIN: Age VS PClass")
plt.show()
print(train_data.head())
#imputation function
# Define the function to impute age
def impute_age(row):
    Age = row['Age']  # Access 'Age' column value from the current row
    Pclass = row['Pclass']  # Access 'Pclass' column value from the current row
    
    # Impute age based on Pclass if Age is NaN
    if pd.isnull(Age):
        if Pclass == 1:
            return 37  # Assume average age for Pclass 1
        elif Pclass == 2:
            return 29  # Assume average age for Pclass 2
        else:
            return 24  # Assume average age for Pclass 3
    else:
        return Age

# Apply the function to the DataFrame row-wise (for each row)
train_data['Age'] = train_data.apply(impute_age, axis=1)
print("Age imputed")

print(train_data.info())
#Dropping missing values
train_data.dropna (inplace = True)

objcat=['Sex', "Embarked"]
for colname in objcat:
    train_data[colname]=train_data[colname].astype('category')
train_data.describe()
train_data.shape
train_data.select_dtypes("category").columns
sex=pd.get_dummies(train_data['Sex'], drop_first = True)
embarked=pd.get_dummies(train_data['Embarked'], drop_first= True)
train_data=pd.concat([train_data, sex, embarked], axis = 1)
print("Training_data:" , train_data.shape)
print(train_data.head(5))
print(train_data.info())
# After one-hot encoding
train_data = pd.concat([train_data, sex, embarked], axis=1)
train_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)  # <- THIS IS IMPORTANT
# Save preprocessed train data for ML
train_data.to_csv("train_processed.csv", index=False)
print("Preprocessed train data saved as train_processed.csv")
