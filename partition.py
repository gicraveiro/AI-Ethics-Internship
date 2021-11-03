# importing the dataset
import pandas as pd
import matplotlib.pyplot as plt
housing = pd.read_csv("data/Facebook/Privacy/Annotated/DataPolicy.csv")
# looking at the dataset
print(housing.head())
print(housing.info())
#print(housing.ocean_proximity.value_counts())
#print(housing.describe())

#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)