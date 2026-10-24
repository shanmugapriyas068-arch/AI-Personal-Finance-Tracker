import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Sample synthetic training data
data = pd.DataFrame({
    "Amount": [
        100, 150, 200, 300, 500,
        1000, 1500, 2000, 2500, 3000,
        500, 1000, 200, 1500, 3000
    ],
    "Category": [
        "Food", "Food", "Food", "Food", "Food",
        "Shopping", "Shopping", "Shopping",
        "Shopping", "Shopping",
        "Travel", "Travel", "Travel",
        "Travel", "Travel"
    ]
})

X = data[["Amount"]]
y = data["Category"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("AI Expense Classification Model Trained!")
print("Sample prediction:", model.predict(pd.DataFrame({"Amount": [1500]}))[0])