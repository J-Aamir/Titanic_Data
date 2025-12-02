import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import warnings
plt.rcParams["figure.figsize"] = [10,15]
#matplotlib inline
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category='FutureWarning')
#importing file
test_data=pd.read_csv ("test.csv")

#TEST DATA EDA
print("TEST DATA")
print(test_data.head(5))
test_data=test_data.drop(["Cabin", "PassengerId", "Name", "Ticket"], axis=1)
print(test_data.info())
print("test_data:" , test_data.shape)

#Checking missing data by heatmap
sns.heatmap(test_data.isnull(), yticklabels=False, cbar=False,cmap="tab20c_r")
plt.title("Missing Data of testing set")
plt.show()

#Draw boxplot of age to fill according to the gender
plt.figure(figsize=(10,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = test_data). set_title("Age VS PClass")
plt.show()
print(test_data.head())
#imputation function
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 25
    else:
        return Age
test_data['Age'] = test_data.apply(lambda row: impute_age([row['Age'], row['Pclass']]), axis=1)  
print(test_data.info())
#Dropping missing values
test_data.dropna (inplace = True)

objcat=['Sex', "Embarked"]
for colname in objcat:
    test_data[colname]=test_data[colname].astype('category')
test_data.describe()
test_data.shape
test_data.select_dtypes("category").columns
sex=pd.get_dummies(test_data['Sex'], drop_first = True)
embarked=pd.get_dummies(test_data['Embarked'], drop_first= True)
test_data=pd.concat([test_data, sex, embarked], axis = 1)
print("Testing_data:" , test_data.shape)
print(test_data.head(5))
test_data = pd.concat([test_data, sex_test, embarked_test], axis=1)
test_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)  # <- DROP originals
# Save preprocessed test data for ML
test_data.to_csv("test_processed.csv", index=False)
print("Preprocessed test data saved as test_processed.csv")
