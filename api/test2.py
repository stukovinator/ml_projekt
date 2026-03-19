import pickle

with open("model/model.pkl", "rb") as f:
    data = pickle.load(f)

print(data)