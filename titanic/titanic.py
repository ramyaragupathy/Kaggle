import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(28)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
titanic = pd.read_csv('../input/train.csv')
titanic.head(3)
titanic.info()
titanic["Sex"].value_counts()
titanic["Embarked"].value_counts()

# encode categorical attributes
titanic["Sex_enc"] = titanic["Sex"].map({"female": 1, "male": 0}) # encode sex
titanic["Embarked_enc"] = titanic["Embarked"].map({"S": 0, "C": 1, "Q": 2}) # encode Embarked

# %matplotlib inline
titanic.hist(bins=50, figsize=(20, 15))
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16, 12))
attr_ls = ["Sex", "Parch", "Pclass", "SibSp", "Embarked"]
for attr, ax in zip(attr_ls[:3], axes[0]):
    titanic.pivot_table("PassengerId", attr, "Survived", "count").plot(kind="bar", stacked=True, ax=ax)
for attr, ax in zip(attr_ls[3:], axes[1]):
    titanic.pivot_table("PassengerId", attr, "Survived", "count").plot(kind="bar", stacked=True, ax=ax)

    corr = titanic[["Survived", "Sex_enc", "Parch", "Pclass", "SibSp", "Embarked_enc"]].corr()
    np.abs(corr["Survived"]).sort_values(ascending=False)
