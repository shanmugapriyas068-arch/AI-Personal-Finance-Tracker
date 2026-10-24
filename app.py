import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
st.set_page_config(
    page_title="AI Personal Finance Tracker",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI Personal Finance Tracker")
# Train AI model using synthetic sample data
training_data = pd.DataFrame({
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

ai_model = DecisionTreeClassifier(random_state=42)
ai_model.fit(training_data[["Amount"]], training_data["Category"])
st.write("Track your money. Understand your spending. Save smarter!")

st.sidebar.header("Financial Details")

income = st.sidebar.number_input(
    "Monthly Income (₹)",
    min_value=0,
    value=20000,
    step=1000
)

st.sidebar.subheader("Add Your Expenses")

if "expenses" not in st.session_state:
    st.session_state.expenses = []

category = st.sidebar.selectbox(
    "Expense Category",
    ["Food", "Travel", "Education", "Shopping", "Bills", "Others"]
)

amount = st.sidebar.number_input(
    "Expense Amount (₹)",
    min_value=0,
    value=0,
    step=100
)

if st.sidebar.button("Add Expense"):
    if amount > 0:
        st.session_state.expenses.append({
            "Category": category,
            "Amount": amount
        })
        st.sidebar.success("Expense added!")
    else:
        st.sidebar.warning("Enter an amount greater than zero.")

expense_data = pd.DataFrame(
    st.session_state.expenses,
    columns=["Category", "Amount"]
)

total_expenses = int(expense_data["Amount"].sum())
savings = income - total_expenses

col1, col2, col3 = st.columns(3)

col1.metric("Monthly Income", f"₹{income:,.0f}")
col2.metric("Total Expenses", f"₹{total_expenses:,.0f}")
col3.metric("Remaining Balance", f"₹{savings:,.0f}")
st.subheader("📥 Download Expense Report")

if not expense_data.empty:
    csv_data = expense_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Expenses as CSV",
        data=csv_data,
        file_name="my_expenses.csv",
        mime="text/csv"
    )
else:
    st.info("Add an expense to enable the download.")
st.subheader("📋 Your Expense Records")

if not expense_data.empty:
    st.dataframe(expense_data, use_container_width=True)

    category_totals = (
        expense_data.groupby("Category", as_index=False)["Amount"].sum()
    )

    st.subheader("📊 Expense Distribution")

    fig = px.pie(
        category_totals,
        names="Category",
        values="Amount",
        title="Your Expenses by Category"
    )

    st.plotly_chart(fig, use_container_width=True)
    bar_fig = px.bar(
        category_totals,
        x="Category",
        y="Amount",
        title="Category-wise Expense Comparison",
        labels={
            "Category": "Expense Category",
            "Amount": "Amount (₹)"
        },
        text="Amount"
    )

    st.plotly_chart(bar_fig, use_container_width=True)
    if st.button("Clear All Expenses"):
        st.session_state.expenses = []
        st.rerun()
else:
    st.info("No expenses added yet. Use the sidebar to add your first expense.")

st.subheader("💡 Financial Insight")
st.subheader("🤖 AI Expense Category Prediction")

prediction_amount = st.number_input(
    "Enter an amount for AI prediction (₹)",
    min_value=1,
    value=1500,
    step=100,
    key="prediction_amount"
)

if st.button("Predict Expense Category"):
    prediction = ai_model.predict(
        pd.DataFrame({"Amount": [prediction_amount]})
    )[0]

    st.info(f"AI predicted category: {prediction}")

    st.caption(
        "Demo prediction using synthetic training data. "
        "Amount alone cannot reliably identify a real expense category."
    )
if income > 0:
    savings_rate = (savings / income) * 100
    st.write(f"Your remaining balance is ₹{savings:,.0f}.")
    st.write(f"Your savings rate is {savings_rate:.1f}%.")

    if savings > 0:
        st.success("Your income is higher than your expenses!")
    elif savings == 0:
        st.info("Your income and expenses are equal.")
    else:
        st.warning("Your expenses exceed your income.")
else:
    st.info("Enter your monthly income to calculate your financial insights.")