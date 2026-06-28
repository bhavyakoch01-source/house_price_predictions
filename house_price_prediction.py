import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data = {
    "Area": [1000, 1200, 1500, 1800, 2000],
    "Bedrooms": [2, 2, 3, 3, 4],
    "Bathrooms": [1, 2, 2, 3, 3],
    "Price": [3000000, 3600000, 4500000, 5400000, 6000000]
}

df = pd.DataFrame(data)

X = df[["Area", "Bedrooms", "Bathrooms"]]

y = df["Price"]

model = DecisionTreeRegressor()
model.fit(X, y)

print("Model trained successfully!")

print("\nEnter details of the house:")

area = int(input("Area (sq ft): "))
bedrooms = int(input("Number of Bedrooms: "))
bathrooms = int(input("Number of Bathrooms: "))

prediction = model.predict([[area, bedrooms, bathrooms]])

print(f"\nPredicted House Price: ₹{prediction[0]:,.0f}")
