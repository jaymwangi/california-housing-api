import pickle

MODEL_PATH = "C:/Users/jaymw/OneDrive/Documents/Jupyter Notebook/AI and DATA/Projects/USA_California_Housing_Price_Prediction/model/model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print(type(model))  # Should print something like <class 'sklearn.linear_model._base.LinearRegression'>
