from sklearn.linear_model import LinearRegression
import numpy as np
import sqlite3
import streamlit as st
import pandas as pd

# Page title
st.title("AI Business Analyst Copilot")

st.caption(
    "Upload your sales data and receive automated analytics, SQL insights, and machine learning predictions."
)

# File upload
uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("data/sales.csv")
    
required_columns = ["Region", "Category", "Sales", "Profit"]

missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.write("Your CSV must contain these columns:")
    st.write(required_columns)
    st.stop()

st.divider()

# Dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

st.subheader("📌 Dataset Information")
st.write("Rows:", data.shape[0])
st.write("Columns:", data.shape[1])

st.subheader("📋 Data Summary")
st.write(data.describe())

st.divider()

# SQL section
conn = sqlite3.connect("sales_data.db")

data.to_sql("sales", conn, if_exists="replace", index=False)

sql_query = """
SELECT Region, SUM(Sales) AS Total_Sales
FROM sales
GROUP BY Region
ORDER BY Total_Sales DESC
"""

sql_result = pd.read_sql(sql_query, conn)

st.subheader("🗄️ SQL Result: Sales by Region")
st.dataframe(sql_result)

st.divider()

# KPI section
total_sales = data["Sales"].sum()
total_profit = data["Profit"].sum()
average_sales = data["Sales"].mean()
highest_sale = data["Sales"].max()
lowest_sale = data["Sales"].min()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Sales", total_sales)

with col2:
    st.metric("Total Profit", total_profit)

with col3:
    st.metric("Average Sales", round(average_sales, 2))

col4, col5 = st.columns(2)

with col4:
    st.metric("Highest Sale", highest_sale)

with col5:
    st.metric("Lowest Sale", lowest_sale)

st.divider()

# Charts
sales_by_region = data.groupby("Region")["Sales"].sum()

st.subheader("📈 Sales by Region")
st.bar_chart(sales_by_region)

profit_by_category = data.groupby("Category")["Profit"].sum()

st.subheader("💰 Profit by Category")
st.bar_chart(profit_by_category)

st.divider()

# AI insights
best_region = data.groupby("Region")["Sales"].sum().idxmax()
worst_region = data.groupby("Region")["Sales"].sum().idxmin()
best_category = data.groupby("Category")["Profit"].sum().idxmax()

st.subheader("🤖 AI Business Insights")

st.write(f"🏆 Best Region: {best_region}")
st.write(f"⚠️ Improvement Needed: {worst_region}")
st.write(f"💰 Most Profitable Category: {best_category}")

if total_profit > 5000:
    ai_message = (
        f"Business performance is excellent. "
        f"{best_category} is driving profitability. "
        f"Continue investing in {best_region} while improving sales in {worst_region}."
    )
    st.success(ai_message)

elif total_profit > 2000:
    ai_message = (
        f"Business performance is good. "
        f"{best_category} is the strongest category. "
        f"Focus on increasing sales in {worst_region}."
    )
    st.warning(ai_message)

else:
    ai_message = (
        f"Profitability needs improvement. "
        f"Review pricing strategy and focus on {worst_region}. "
        f"Leverage strengths from {best_category}."
    )
    st.error(ai_message)

st.divider()

# ML prediction
st.subheader("🔮 Sales Prediction")

X = np.array(range(len(data))).reshape(-1, 1)
y = data["Sales"]

model = LinearRegression()
model.fit(X, y)

next_sale = model.predict([[len(data)]])
predicted_sale = round(next_sale[0], 2)

st.metric("Predicted Next Sale", predicted_sale)

# Report text
report = f"""
AI Business Analyst Copilot Report

Dataset Information:
Rows: {data.shape[0]}
Columns: {data.shape[1]}

Key Metrics:
Total Sales: {total_sales}
Total Profit: {total_profit}
Average Sales: {round(average_sales, 2)}
Highest Sale: {highest_sale}
Lowest Sale: {lowest_sale}

Business Insights:
Best Region: {best_region}
Improvement Needed: {worst_region}
Most Profitable Category: {best_category}

AI Recommendation:
{ai_message}

Machine Learning Prediction:
Predicted Next Sale: {predicted_sale}
"""

st.divider()

# Download report
st.subheader("📥 Download Report")

st.download_button(
    label="Download Business Report",
    data=report,
    file_name="business_report.txt",
    mime="text/plain"
)

conn.close()