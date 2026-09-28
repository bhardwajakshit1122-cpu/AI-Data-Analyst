from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)
def calculate_average(numbers):
    total=sum(numbers)
    average=total/len(numbers)
    return average

sales = [1000,1500,2000,2500]
result=calculate_average(sales)
print("Average sales :",result)

print("===============================")
print("    AI DATA ANALYST      ")
print("===============================")
print("project status: starting...")
print("data analysis engine:preparing...")
print("AI Analyst: coming soon...")

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("data/sales.csv")

print("----FIRST 5 ROWS----")
print(data.head())

print("\n----DATA SHAPE----")
print(data.shape)

print("\n----COLUMN NAMES----")
print(data.columns)

print("\n----DATA INFORMATION----")
print(data.info())

print("\n----DATA QUALITY CHECK----")

print("ROWS:",data.shape[0])
print("COLUMNS:", data.shape[1])
print("\nMissing Values:")
print(data.isnull().sum())
print("\nDuplicate Rows:",data.duplicated().sum())
print("\n------NUMERICAL ANALYSIS-----")
numeric_columns=data.select_dtypes(include="number").columns
print(data[numeric_columns].describe())
print("\n-----AUTOMATIC COLOUM DETECTION-----")
numeric_columns=data.select_dtypes(include="number").columns
print("Numerical Columns:")
print(list(numeric_columns))
print("\n-----CATEGORICAL ANALYSIS-----")
categorical_columns=data.select_dtypes(include="object").columns
print("categorical Columns")
print(list(categorical_columns))
for column in categorical_columns:print(f"\n{column}-Unique Values:")
print(data[column].unique)
print("\n-----AUTOMATIC BUSINESS INSIGHTS-----")
product_sales=data.groupby("Product")["Quantity"].sum()
best_product = product_sales.idxmax()
highest_quantity = product_sales.max()

print("Product-wise Quantity:")
print(product_sales)
print("\nTop Selling Product:",best_product)
print("Quantity Sold:",highest_quantity) 
print("\n-----REVENUE & PROFIT ANALYSIS-----")
data["Revenue"]=data["Quantity"]*data["Price"]
data["Total_Cost"]=data["Quantity"]*data["Cost"]
data["Profit"]=data["Revenue"]-data["Total_Cost"]

print("Total Revenue:",data["Revenue"].sum())
print("Total Cost:",data["Total_Cost"].sum())
print("Total Profit:",data["Profit"].sum())
print("\n-----PRODUCT-WISE REVENUE & PROFIT-----")
product_performance=data.groupby("Product")[["Revenue","Profit"]].sum()
print(product_performance)
print("\n-----PROFIT INSIGHT-----")
most_profitable_product=product_performance["Profit"].idxmax()
highest_profit=product_performance["Profit"].max()
print("Most Profitable Product:",most_profitable_product)
print("Highest Profit:",highest_profit)
print("\n----- REGION-WISE ANALYSIS -----")

region_performance = data.groupby("Region")[[
    "Quantity",
    "Revenue",
    "Profit"
]].sum()

print(region_performance)
print("\n----- REGION INSIGHTS -----")

highest_revenue_region = region_performance["Revenue"].idxmax()
highest_revenue = region_performance["Revenue"].max()

highest_profit_region = region_performance["Profit"].idxmax()
highest_profit = region_performance["Profit"].max()

highest_quantity_region = region_performance["Quantity"].idxmax()
highest_quantity = region_performance["Quantity"].max()

print("Highest Revenue Region:", highest_revenue_region)
print("Revenue:", highest_revenue)

print("\nHighest Profit Region:", highest_profit_region)
print("Profit:", highest_profit)

print("\nHighest Quantity Region:", highest_quantity_region)
print("Quantity:", highest_quantity)
print("\n----- DATE ANALYSIS -----")

data["Date"] = pd.to_datetime(data["Date"])

print("Date Data Type:", data["Date"].dtype)

print("\nDate Range:")
print("Start Date:", data["Date"].min())
print("End Date:", data["Date"].max())
data["Month"] = data["Date"].dt.to_period("M")

monthly_revenue = data.groupby("Month")["Revenue"].sum()

print("\n----- MONTHLY REVENUE -----")
print(monthly_revenue)
print("\n----- MONTHLY REVENUE CHART -----")

monthly_revenue.plot(
    kind="line",
    marker="o",
    title="Monthly Revenue"
)

plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()
product_revenue_profit=data.groupby("Product")[["Revenue","Profit"]].sum()
print(product_revenue_profit)
product_revenue_profit.plot(kind="bar",title="Product-wise Revenue and Profit")
plt.xlabel("Product")
plt.ylabel("Amount")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()          
region_revenue_profit=data.groupby("Region")[["Revenue","Profit"]].sum()
print(region_revenue_profit)
region_revenue_profit.plot(
    kind="bar",
    title="Region-wise Revenue and Profit"
)

plt.xlabel("Region")
plt.ylabel("Amount")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()
numeric_columns=data.select_dtypes(include="number").columns
print("Numerical columns:")
print(numeric_columns)
print("\nAutomated Numerical Analysis:")
print(data[numeric_columns].describe)
print("\nAutomated Missing Value Analysis:")
missing_values=data.isnull().sum()
print(missing_values)
print("\nAutomated Duplicate Analysis:")
duplicate_count=data.duplicated().sum()
print("Duplicate Rows:",duplicate_count)
if duplicate_count==0:
    print("No duplicate rows found.")
else:
    print("Duplicate rows detected.")
print("\nAutomated Categorical Analysis:")

categorical_columns = data.select_dtypes(include="object").columns

print("Categorical Columns:")
print(categorical_columns)
for column in categorical_columns:
    print(f"\n{column} Value Counts:")
    print(data[column].value_counts())
print("\nAutomated Outlier Analysis:")

for column in numeric_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    print(f"{column}: {len(outliers)} outliers")
    print("\nAutomated Correlation Analysis:")

correlation_matrix = data[numeric_columns].corr()

print(correlation_matrix)
plt.figure(figsize=(8, 6))

plt.imshow(correlation_matrix, cmap="coolwarm")

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()
correlation_pairs = correlation_matrix.unstack()

correlation_pairs = correlation_pairs[
    correlation_pairs.index.get_level_values(0) !=
    correlation_pairs.index.get_level_values(1)
]

strongest_pair = correlation_pairs.abs().idxmax()

print("\nStrongest Correlation:")
print(strongest_pair)
print("Correlation Value:", correlation_pairs[strongest_pair])
print("\nAutomated Business Insights:")

highest_revenue_product = product_revenue_profit["Revenue"].idxmax()
highest_profit_product = product_revenue_profit["Profit"].idxmax()

print(
    f"Highest revenue product: {highest_revenue_product}"
)

print(
    f"Highest profit product: {highest_profit_product}"
)
highest_revenue_region = region_revenue_profit["Revenue"].idxmax()
highest_profit_region = region_revenue_profit["Profit"].idxmax()

print(
    f"Highest revenue region: {highest_revenue_region}"
)

print(
    f"Highest profit region: {highest_profit_region}"
)
highest_product_revenue = product_revenue_profit.loc[
    highest_revenue_product, "Revenue"
]

highest_product_profit = product_revenue_profit.loc[
    highest_profit_product, "Profit"
]

print(
    f"{highest_revenue_product} generated the highest revenue of ₹{highest_product_revenue:,.0f}."
)

print(
    f"{highest_profit_product} generated the highest profit of ₹{highest_product_profit:,.0f}."
)
highest_region_revenue = region_revenue_profit.loc[
    highest_revenue_region, "Revenue"
]

highest_region_profit = region_revenue_profit.loc[
    highest_profit_region, "Profit"
]

print(
    f"{highest_revenue_region} generated the highest regional revenue of ₹{highest_region_revenue:,.0f}."
)

print(
    f"{highest_profit_region} generated the highest regional profit of ₹{highest_region_profit:,.0f}."
)
highest_region_revenue = region_revenue_profit.loc[
    highest_revenue_region, "Revenue"
]

highest_region_profit = region_revenue_profit.loc[
    highest_profit_region, "Profit"
]

print(
    f"{highest_revenue_region} generated the highest regional revenue of ₹{highest_region_revenue:,.0f}."
)

print(
    f"{highest_profit_region} generated the highest regional profit of ₹{highest_region_profit:,.0f}."
)
region_revenue_profit["Profit_Margin"] = (
    region_revenue_profit["Profit"]
    / region_revenue_profit["Revenue"]
) * 100

print("\nRegion-wise Profit Margin:")
print(region_revenue_profit)
highest_margin_region = region_revenue_profit["Profit_Margin"].idxmax()

highest_region_margin = region_revenue_profit.loc[
    highest_margin_region, "Profit_Margin"
]

print(
    f"Highest profit margin region: "
    f"{highest_margin_region} ({highest_region_margin:.2f}%)"
)
print("\nProduct-wise Profit Margin:")

product_revenue_profit["Profit_Margin"] = (
    product_revenue_profit["Profit"]
    / product_revenue_profit["Revenue"]
) * 100

print(product_revenue_profit)

highest_margin_product = product_revenue_profit["Profit_Margin"].idxmax()

highest_margin = product_revenue_profit.loc[
    highest_margin_product, "Profit_Margin"
]

print(
    f"Highest profit margin product: "
    f"{highest_margin_product} ({highest_margin:.2f}%)"
)
print("\nOverall Business Summary:")

total_revenue = data["Revenue"].sum()
total_cost = data["Total_Cost"].sum()
total_profit = data["Profit"].sum()

overall_profit_margin = (
    total_profit / total_revenue
) * 100

print(f"Total Revenue: ₹{total_revenue:,.0f}")
print(f"Total Cost: ₹{total_cost:,.0f}")
print(f"Total Profit: ₹{total_profit:,.0f}")
print(f"Overall Profit Margin: {overall_profit_margin:.2f}%")
print("\nAverage Order Value:")

average_order_value = data["Revenue"].mean()

print(f"Average Revenue per Order: ₹{average_order_value:,.2f}")
print("\nProduct Revenue Contribution:")

product_revenue_profit["Revenue_Contribution"] = (
    product_revenue_profit["Revenue"]
    / total_revenue
) * 100

print(
    product_revenue_profit[
        ["Revenue", "Revenue_Contribution"]
    ]
)
# ============================================================
# STEP 29 - TOP REVENUE CONTRIBUTING PRODUCT
# ============================================================

print("\nTop Revenue Contributing Product:")

top_revenue_product = product_revenue_profit["Revenue"].idxmax()
top_revenue_contribution = product_revenue_profit.loc[
    top_revenue_product, "Revenue_Contribution"
]

print(
    f"{top_revenue_product} contributes "
    f"{top_revenue_contribution:.2f}% of total revenue."
)


# ============================================================
# STEP 30 - MONTHLY REVENUE GROWTH
# ============================================================

print("\nMonthly Revenue Growth:")

monthly_revenue_growth = monthly_revenue.pct_change() * 100

print(monthly_revenue_growth)


# ============================================================
# STEP 31 - BEST REVENUE MONTH
# ============================================================

best_revenue_month = monthly_revenue.idxmax()
best_month_revenue = monthly_revenue.max()

print(
    f"Best revenue month: {best_revenue_month}"
)

print(
    f"Revenue generated: ₹{best_month_revenue:,.0f}"
)


# ============================================================
# STEP 32 - LOWEST REVENUE MONTH
# ============================================================

lowest_revenue_month = monthly_revenue.idxmin()
lowest_month_revenue = monthly_revenue.min()

print(
    f"Lowest revenue month: {lowest_revenue_month}"
)

print(
    f"Revenue generated: ₹{lowest_month_revenue:,.0f}"
)


# ============================================================
# STEP 33 - AUTOMATED ANOMALY DETECTION
# ============================================================

print("\nAutomated Anomaly Detection:")

revenue_Q1 = data["Revenue"].quantile(0.25)
revenue_Q3 = data["Revenue"].quantile(0.75)

revenue_IQR = revenue_Q3 - revenue_Q1

revenue_lower_bound = revenue_Q1 - 1.5 * revenue_IQR
revenue_upper_bound = revenue_Q3 + 1.5 * revenue_IQR

revenue_anomalies = data[
    (data["Revenue"] < revenue_lower_bound) |
    (data["Revenue"] > revenue_upper_bound)
]

print(
    f"Revenue anomalies detected: "
    f"{len(revenue_anomalies)}"
)

if len(revenue_anomalies) > 0:
    print(revenue_anomalies[
        ["Date", "Product", "Revenue"]
    ])
else:
    print("No revenue anomalies detected.")


# ============================================================
# STEP 34 - HIGH VALUE TRANSACTIONS
# ============================================================

print("\nHigh Value Transactions:")

high_value_threshold = data["Revenue"].quantile(0.90)

high_value_transactions = data[
    data["Revenue"] >= high_value_threshold
]

print(
    f"High-value threshold: "
    f"₹{high_value_threshold:,.0f}"
)

print(
    high_value_transactions[
        ["Date", "Product", "Revenue", "Profit"]
    ]
)


# ============================================================
# STEP 35 - PRODUCT PERFORMANCE SUMMARY
# ============================================================

print("\nProduct Performance Summary:")

product_summary = data.groupby("Product").agg(
    Total_Quantity=("Quantity", "sum"),
    Total_Revenue=("Revenue", "sum"),
    Total_Profit=("Profit", "sum")
)

product_summary["Profit_Margin"] = (
    product_summary["Total_Profit"]
    / product_summary["Total_Revenue"]
) * 100

print(product_summary)


# ============================================================
# STEP 36 - REGION PERFORMANCE SUMMARY
# ============================================================

print("\nRegion Performance Summary:")

region_summary = data.groupby("Region").agg(
    Total_Quantity=("Quantity", "sum"),
    Total_Revenue=("Revenue", "sum"),
    Total_Profit=("Profit", "sum")
)

region_summary["Profit_Margin"] = (
    region_summary["Total_Profit"]
    / region_summary["Total_Revenue"]
) * 100

print(region_summary)


# ============================================================
# STEP 37 - BEST PROFIT MARGIN PRODUCT
# ============================================================

best_margin_product = product_summary["Profit_Margin"].idxmax()
best_margin_value = product_summary["Profit_Margin"].max()

print(
    f"\nBest profit margin product: "
    f"{best_margin_product}"
)

print(
    f"Profit Margin: {best_margin_value:.2f}%"
)


# ============================================================
# STEP 38 - BEST PROFIT MARGIN REGION
# ============================================================

best_margin_region = region_summary["Profit_Margin"].idxmax()
best_region_margin_value = region_summary["Profit_Margin"].max()

print(
    f"\nBest profit margin region: "
    f"{best_margin_region}"
)

print(
    f"Profit Margin: {best_region_margin_value:.2f}%"
)


# ============================================================
# STEP 39 - AUTOMATED BUSINESS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("AUTOMATED BUSINESS SUMMARY")
print("=" * 60)

print(
    f"Total Revenue: ₹{total_revenue:,.0f}"
)

print(
    f"Total Cost: ₹{total_cost:,.0f}"
)

print(
    f"Total Profit: ₹{total_profit:,.0f}"
)

print(
    f"Overall Profit Margin: "
    f"{overall_profit_margin:.2f}%"
)

print(
    f"Top Revenue Product: "
    f"{top_revenue_product}"
)

print(
    f"Top Revenue Region: "
    f"{highest_revenue_region}"
)

print(
    f"Best Profit Margin Product: "
    f"{best_margin_product}"
)

print(
    f"Best Profit Margin Region: "
    f"{best_margin_region}"
)

print(
    f"Best Revenue Month: "
    f"{best_revenue_month}"
)

print(
    f"Revenue Anomalies: "
    f"{len(revenue_anomalies)}"
)

print("=" * 60)


# ============================================================
# STEP 40 - AUTOMATED TEXT INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("AI DATA ANALYST - AUTOMATED INSIGHTS")
print("=" * 60)

print(
    f"1. {top_revenue_product} is the highest revenue-generating product."
)

print(
    f"2. {highest_revenue_region} is the highest revenue-generating region."
)

print(
    f"3. {best_margin_product} has the highest product profit margin "
    f"of {best_margin_value:.2f}%."
)

print(
    f"4. {best_margin_region} has the highest regional profit margin "
    f"of {best_region_margin_value:.2f}%."
)

print(
    f"5. {best_revenue_month} generated the highest monthly revenue "
    f"of ₹{best_month_revenue:,.0f}."
)

print(
    f"6. The overall business profit margin is "
    f"{overall_profit_margin:.2f}%."
)

print(
    f"7. {len(revenue_anomalies)} revenue anomalies were detected."
)

print("=" * 60)
data["Date"]=pd.to_datetime(data["Date"])
print("\n---Forecasting Data Preparation---")
print(data[["Date","Revenue"]].head())
monthly_revenue=(
      data
      .set_index("Date")
      .resample("ME")["Revenue"]
      .sum()
      .reset_index()
                 )
print("\n--Monthly Revenue For Forecasting---")
print(monthly_revenue)
# Monthly Revenue Trend

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_revenue['Date'],
    monthly_revenue['Revenue'],
    marker='o'
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()
# ============================================================
# STEP 41.4: Prepare Data for Forecasting
# ============================================================

print("\n--- Preparing Data for Forecasting ---")

# Create a numerical time index
monthly_revenue['Month_Index'] = range(len(monthly_revenue))

print(monthly_revenue)


# ============================================================
# STEP 41.5: Check Whether Enough Data Is Available
# ============================================================

if len(monthly_revenue) < 4:

    print("\nNot enough monthly data for reliable forecasting.")
    print("At least 4 months of data are required for this demo model.")

else:

    # ========================================================
    # STEP 41.6: Train Forecasting Model
    # ========================================================

    from sklearn.linear_model import LinearRegression

    # Feature
    X = monthly_revenue[['Month_Index']]

    # Target
    y = monthly_revenue['Revenue']

    # Use earlier months for training
    train_size = int(len(monthly_revenue) * 0.8)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # Create model
    forecast_model = LinearRegression()

    # Train model
    forecast_model.fit(X_train, y_train)

    print("\n--- Forecasting Model Trained ---")

    # Test prediction
    test_predictions = forecast_model.predict(X_test)

    print("\nActual Revenue:")
    print(y_test.values)

    print("\nPredicted Revenue:")
    print(test_predictions)


    # ========================================================
    # STEP 41.7: Forecast Future Revenue
    # ========================================================

    future_months = 3

    last_month_index = monthly_revenue['Month_Index'].iloc[-1]

    future_indices = range(
        last_month_index + 1,
        last_month_index + 1 + future_months
    )

    future_predictions = forecast_model.predict(
        [[index] for index in future_indices]
    )

    print("\n--- Future Revenue Forecast ---")

    for i, prediction in enumerate(future_predictions, start=1):
        print(
            f"Future Month {i}: "
            f"Estimated Revenue = {prediction:.2f}"
        )


    # ========================================================
    # STEP 41.8: Create Forecast Summary
    # ========================================================

    forecast_summary = pd.DataFrame({
        'Future_Month': [f'Month + {i}' for i in range(1, future_months + 1)],
        'Predicted_Revenue': future_predictions
    })

    print("\n--- Forecast Summary ---")
    print(forecast_summary)


    # ========================================================
    # Forecast Visualization
    # ========================================================

    plt.figure(figsize=(10, 5))

    # Historical revenue
    plt.plot(
        monthly_revenue['Date'],
        monthly_revenue['Revenue'],
        marker='o',
        label='Historical Revenue'
    )

    # Create future dates
    future_dates = pd.date_range(
        start=monthly_revenue['Date'].iloc[-1] + pd.offsets.MonthEnd(1),
        periods=future_months,
        freq='ME'
    )

    # Forecast revenue
    plt.plot(
        future_dates,
        future_predictions,
        marker='o',
        linestyle='--',
        label='Forecast Revenue'
    )

    plt.title("Revenue Forecast")
    plt.xlabel("Month")
    plt.ylabel("Revenue")

    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
# ============================================================
# STEP 42: Automated Forecast Insights
# ============================================================

if len(monthly_revenue) >= 4:

    # Historical average revenue
    historical_avg = monthly_revenue['Revenue'].mean()

    # Forecast average revenue
    forecast_avg = forecast_summary['Predicted_Revenue'].mean()

    # Forecast change percentage
    forecast_change = (
        (forecast_avg - historical_avg)
        / historical_avg
    ) * 100

    print("\n--- Automated Forecast Insights ---")

    print(f"Historical Average Monthly Revenue: {historical_avg:.2f}")

    print(f"Average Forecasted Revenue: {forecast_avg:.2f}")

    print(f"Forecast Change: {forecast_change:.2f}%")

    # Determine overall trend
    if forecast_change > 0:
        forecast_trend = "Increasing"

    elif forecast_change < 0:
        forecast_trend = "Decreasing"

    else:
        forecast_trend = "Stable"

    print(f"Forecast Trend: {forecast_trend}")

    # Generate business insight
    if forecast_change > 5:

        forecast_insight = (
            "Revenue is projected to increase significantly "
            "compared with the historical monthly average."
        )

    elif forecast_change > 0:

        forecast_insight = (
            "Revenue is projected to increase slightly "
            "compared with the historical monthly average."
        )

    elif forecast_change < -5:

        forecast_insight = (
            "Revenue is projected to decrease significantly "
            "compared with the historical monthly average."
        )

    elif forecast_change < 0:

        forecast_insight = (
            "Revenue is projected to decrease slightly "
            "compared with the historical monthly average."
        )

    else:

        forecast_insight = (
            "Revenue is projected to remain relatively stable "
            "compared with the historical monthly average."
        )

    print(f"\nBusiness Insight: {forecast_insight}")

else:

    print("\nForecast insights skipped because there is not enough data.")
# ============================================================
# STEP 43: AI ANALYSIS CONTEXT
# ============================================================

print("\n--- AI Analysis Context ---")

# Basic dataset information
total_rows = len(data)
total_columns = len(data.columns)

# Revenue information
total_revenue = data['Revenue'].sum()
average_revenue = data['Revenue'].mean()

# Profit information
total_profit = data['Profit'].sum()
average_profit = data['Profit'].mean()

# Top product by revenue
top_product = (
    data.groupby('Product')['Revenue']
    .sum()
    .idxmax()
)

top_product_revenue = (
    data.groupby('Product')['Revenue']
    .sum()
    .max()
)

# Top region by revenue
top_region = (
    data.groupby('Region')['Revenue']
    .sum()
    .idxmax()
)

top_region_revenue = (
    data.groupby('Region')['Revenue']
    .sum()
    .max()
)

# Create structured context
ai_context = {
    "dataset": {
        "rows": total_rows,
        "columns": total_columns
    },

    "revenue": {
        "total": total_revenue,
        "average_transaction": average_revenue
    },

    "profit": {
        "total": total_profit,
        "average_transaction": average_profit
    },

    "top_product": {
        "name": top_product,
        "revenue": top_product_revenue
    },

    "top_region": {
        "name": top_region,
        "revenue": top_region_revenue
    }
}

print(ai_context)
# ============================================================
# STEP 44: Natural Language Business Summary
# ============================================================

print("\n--- AI-Ready Business Summary ---")

business_summary = f"""
The dataset contains {total_rows} transactions
across {total_columns} columns.

Total revenue is {total_revenue:.2f},
while total profit is {total_profit:.2f}.

The average transaction revenue is {average_revenue:.2f},
and the average transaction profit is {average_profit:.2f}.

The highest revenue-generating product is
{top_product}, generating {top_product_revenue:.2f} in revenue.

The highest revenue-generating region is
{top_region}, generating {top_region_revenue:.2f} in revenue.
"""

print(business_summary)
# ============================================================
# STEP 45: Natural Language Q&A Foundation
# ============================================================

print("\n--- Natural Language Q&A ---")

question = input("\nAsk a question about your data: ")

question_lower = question.lower()

if "total revenue" in question_lower:

    answer = (
        f"Your total revenue is "
        f"{total_revenue:.2f}."
    )

elif "total profit" in question_lower:

    answer = (
        f"Your total profit is "
        f"{total_profit:.2f}."
    )

elif "average revenue" in question_lower:

    answer = (
        f"The average transaction revenue is "
        f"{average_revenue:.2f}."
    )

elif "average profit" in question_lower:

    answer = (
        f"The average transaction profit is "
        f"{average_profit:.2f}."
    )

elif "top product" in question_lower or "best product" in question_lower:

    answer = (
        f"The highest revenue-generating product is "
        f"{top_product}, with revenue of "
        f"{top_product_revenue:.2f}."
    )

elif "top region" in question_lower or "best region" in question_lower:

    answer = (
        f"The highest revenue-generating region is "
        f"{top_region}, with revenue of "
        f"{top_region_revenue:.2f}."
    )

else:

    answer = (
        "I don't have a predefined answer for that question yet."
    )

print("\nAI Data Analyst:")
print(answer)
# ============================================================
# STEP 46: AI Prompt Layer
# ============================================================

def build_ai_prompt(question, context):
    prompt = f"""
You are an AI Data Analyst.

Answer the user's question using only the provided business data.

Business Data:
{context}

User Question:
{question}

Instructions:
- Use only the provided data.
- Do not invent numbers.
- Keep the answer clear and concise.
- If the data is insufficient, say so.
"""

    return prompt


# Create AI-ready context
ai_prompt = build_ai_prompt(
    question,
    ai_context
)

print("\n--- AI Prompt Preview ---")
print(ai_prompt)
# ============================================================
# STEP 47: AI Response Layer
# ============================================================

def generate_ai_response(question, context):
    prompt = build_ai_prompt(question, context)

    # Temporary response
    response = {
        "question": question,
        "prompt": prompt,
        "status": "AI model integration pending"
    }

    return response


ai_response = generate_ai_response(
    question,
    ai_context
)

print("\n--- AI Response Layer ---")
print(ai_response["status"])
# ============================================================
# STEP 48: Add Forecast Information to AI Context
# ============================================================

if len(monthly_revenue) >= 4:

    ai_context["forecast"] = {
        "forecast_months": future_months,
        "average_forecast_revenue": float(forecast_avg),
        "forecast_change_percent": float(forecast_change),
        "forecast_trend": forecast_trend
    }

else:

    ai_context["forecast"] = {
        "status": "Not enough data for forecasting"
    }


print("\n--- Final AI Context ---")
print(ai_context)

# ============================================================
# STEP 51: ENHANCED AI BUSINESS CONTEXT
# ============================================================

ai_context["business_analysis"] = {
    "total_revenue": float(total_revenue),
    "total_profit": float(total_profit),
    "average_revenue": float(average_revenue),
    "average_profit": float(average_profit),
    "top_product": top_product,
    "top_product_revenue": float(top_product_revenue),
    "top_region": top_region,
    "top_region_revenue": float(top_region_revenue)
}

print("\n--- Enhanced AI Context Ready ---")
print(ai_context)


# ============================================================
# STEP 54: DETAILED BUSINESS ANALYSIS FOR AI
# ============================================================

# Product-wise revenue and profit
product_analysis = (
    data.groupby('Product')[['Revenue', 'Profit']]
    .sum()
    .sort_values('Revenue', ascending=False)
)

# Region-wise revenue and profit
region_analysis = (
    data.groupby('Region')[['Revenue', 'Profit']]
    .sum()
    .sort_values('Revenue', ascending=False)
)

# Add detailed analysis to AI context
ai_context["product_analysis"] = product_analysis.to_dict()
ai_context["region_analysis"] = region_analysis.to_dict()

print("\n--- Detailed AI Business Analysis Ready ---")
print("\nProduct Analysis:")
print(product_analysis)
print("\nRegion Analysis:")
print(region_analysis)


# ============================================================
# LOCAL DATA ANALYSIS FALLBACK
# ============================================================

# ============================================================
# STEP 56: BETTER LOCAL AI ANALYST
# ============================================================

def ask_local_ai(question):

    q = question.lower().strip()
    # --------------------------------------------------------
    # STEP 61: AUTOMATED REPORT REQUEST
    # --------------------------------------------------------

    if (
        "dataset summary" in q
        or "data summary" in q
        or "business report" in q
        or "generate report" in q
        or "full report" in q
    ):

        return generate_business_report()
    # --------------------------------------------------------
    # STEP 62: DATASET INFORMATION
    # --------------------------------------------------------

    elif (
        "dataset information" in q
        or "dataset details" in q
        or "how many rows" in q
        or "how many columns" in q
    ):

        return (
            f"The dataset contains {len(data)} rows "
            f"and {len(data.columns)} columns."
        )


    # --------------------------------------------------------
    # COLUMN INFORMATION
    # --------------------------------------------------------

    elif (
        "columns" in q
        or "column names" in q
        or "what columns" in q
    ):

        column_names = ", ".join(data.columns)

        return (
            f"The dataset contains these columns:\n"
            f"{column_names}"
        )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    elif (
        "missing values" in q
        or "missing data" in q
    ):

        missing_values = data.isnull().sum().sum()

        return (
            f"The dataset contains "
            f"{missing_values} missing values."
        )


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    elif (
        "duplicate" in q
        or "duplicate rows" in q
    ):

        duplicate_rows = data.duplicated().sum()

        return (
            f"The dataset contains "
            f"{duplicate_rows} duplicate rows."
        )

    # --------------------------------------------------------
    # TOTAL REVENUE
    # --------------------------------------------------------

    if "total revenue" in q:

        return (
            f"Your total revenue is "
            f"{total_revenue:.2f}."
        )


    # --------------------------------------------------------
    # TOTAL PROFIT
    # --------------------------------------------------------

    elif "total profit" in q:

        return (
            f"Your total profit is "
            f"{total_profit:.2f}."
        )


    # --------------------------------------------------------
    # TOTAL QUANTITY
    # --------------------------------------------------------

    elif (
        "total quantity" in q
        or "total units" in q
        or "how many units" in q
        or "units sold" in q
    ):

        if "Quantity" in data.columns:

            total_quantity = data["Quantity"].sum()

            return (
                f"The total quantity sold is "
                f"{total_quantity} units."
            )

        else:

            return (
                "Quantity information is not available "
                "in the dataset."
            )


    # --------------------------------------------------------
    # AVERAGE REVENUE
    # --------------------------------------------------------

    elif "average revenue" in q:

        return (
            f"The average transaction revenue is "
            f"{average_revenue:.2f}."
        )


    # --------------------------------------------------------
    # AVERAGE PROFIT
    # --------------------------------------------------------

    elif "average profit" in q:

        return (
            f"The average transaction profit is "
            f"{average_profit:.2f}."
        )


    # --------------------------------------------------------
    # AVERAGE PRICE
    # --------------------------------------------------------

    elif "average price" in q:

        if "Price" in data.columns:

            average_price = data["Price"].mean()

            return (
                f"The average product price is "
                f"{average_price:.2f}."
            )

        else:

            return (
                "Price information is not available "
                "in the dataset."
            )


    # --------------------------------------------------------
    # PRODUCT WITH MOST REVENUE
    # --------------------------------------------------------

    elif (
        "most revenue" in q
        or "highest revenue" in q
        or "top revenue product" in q
        or "top product" in q
        or "best product" in q
    ):

        product_data = (
            data.groupby("Product")["Revenue"]
            .sum()
        )

        product_name = product_data.idxmax()
        product_revenue = product_data.max()

        return (
            f"The product generating the most revenue is "
            f"{product_name}, with revenue of "
            f"{product_revenue:.2f}."
        )


    # --------------------------------------------------------
    # PRODUCT WITH MOST PROFIT
    # --------------------------------------------------------

    elif (
        "most profit" in q
        or "highest profit" in q
        or "profitable product" in q
        or "most profitable product" in q
    ):

        product_data = (
            data.groupby("Product")["Profit"]
            .sum()
        )

        product_name = product_data.idxmax()
        product_profit = product_data.max()

        return (
            f"The product generating the most profit is "
            f"{product_name}, with profit of "
            f"{product_profit:.2f}."
        )


    # --------------------------------------------------------
    # PRODUCT WITH MOST UNITS SOLD
    # --------------------------------------------------------

    elif (
        "most units" in q
        or "most quantity" in q
        or "sold the most" in q
        or "sold most" in q
    ):

        if "Quantity" in data.columns:

            product_data = (
                data.groupby("Product")["Quantity"]
                .sum()
            )

            product_name = product_data.idxmax()
            product_quantity = product_data.max()

            return (
                f"The product with the highest number of "
                f"units sold is {product_name}, with "
                f"{product_quantity} units."
            )

        else:

            return (
                "Quantity information is not available "
                "in the dataset."
            )


    # --------------------------------------------------------
    # REGION WITH MOST REVENUE
    # --------------------------------------------------------

    elif (
        "top region" in q
        or "best region" in q
        or "highest revenue region" in q
        or "region most revenue" in q
    ):

        region_data = (
            data.groupby("Region")["Revenue"]
            .sum()
        )

        region_name = region_data.idxmax()
        region_revenue = region_data.max()

        return (
            f"The highest revenue-generating region is "
            f"{region_name}, with revenue of "
            f"{region_revenue:.2f}."
        )


    # --------------------------------------------------------
    # REGION WITH MOST PROFIT
    # --------------------------------------------------------

    elif (
        "region most profit" in q
        or "highest profit region" in q
        or "most profitable region" in q
        or "region highest profit" in q
    ):

        region_data = (
            data.groupby("Region")["Profit"]
            .sum()
        )

        region_name = region_data.idxmax()
        region_profit = region_data.max()

        return (
            f"The region generating the most profit is "
            f"{region_name}, with profit of "
            f"{region_profit:.2f}."
        )


    # --------------------------------------------------------
    # PRODUCT WITH HIGHEST PROFIT MARGIN
    # --------------------------------------------------------

    elif (
        "highest profit margin product" in q
        or "best profit margin product" in q
        or "product highest margin" in q
    ):

        product_data = (
            data.groupby("Product")[["Revenue", "Profit"]]
            .sum()
        )

        product_data["Profit_Margin"] = (
            product_data["Profit"]
            / product_data["Revenue"]
        ) * 100

        product_name = product_data["Profit_Margin"].idxmax()
        profit_margin = product_data["Profit_Margin"].max()

        return (
            f"The product with the highest profit margin is "
            f"{product_name}, with a profit margin of "
            f"{profit_margin:.2f}%."
        )


    # --------------------------------------------------------
    # REGION WITH HIGHEST PROFIT MARGIN
    # --------------------------------------------------------

    elif (
        "highest profit margin region" in q
        or "best profit margin region" in q
        or "region highest margin" in q
    ):

        region_data = (
            data.groupby("Region")[["Revenue", "Profit"]]
            .sum()
        )

        region_data["Profit_Margin"] = (
            region_data["Profit"]
            / region_data["Revenue"]
        ) * 100

        region_name = region_data["Profit_Margin"].idxmax()
        profit_margin = region_data["Profit_Margin"].max()

        return (
            f"The region with the highest profit margin is "
            f"{region_name}, with a profit margin of "
            f"{profit_margin:.2f}%."
        )
    # --------------------------------------------------------
    # PRODUCT COMPARISON
    # --------------------------------------------------------

    elif "compare" in q and "product" in q:

        products = data["Product"].unique()

        if len(products) >= 2:

            product_1 = products[0]
            product_2 = products[1]

            comparison = (
                data.groupby("Product")[["Revenue", "Profit"]]
                .sum()
            )

            revenue_1 = comparison.loc[product_1, "Revenue"]
            revenue_2 = comparison.loc[product_2, "Revenue"]

            profit_1 = comparison.loc[product_1, "Profit"]
            profit_2 = comparison.loc[product_2, "Profit"]

            if revenue_1 > revenue_2:
                revenue_result = product_1
            elif revenue_2 > revenue_1:
                revenue_result = product_2
            else:
                revenue_result = "Both products have equal revenue"

            if profit_1 > profit_2:
                profit_result = product_1
            elif profit_2 > profit_1:
                profit_result = product_2
            else:
                profit_result = "Both products have equal profit"

            return (
                f"Product comparison:\n\n"
                f"{product_1} Revenue: {revenue_1:.2f}\n"
                f"{product_2} Revenue: {revenue_2:.2f}\n\n"
                f"{product_1} Profit: {profit_1:.2f}\n"
                f"{product_2} Profit: {profit_2:.2f}\n\n"
                f"Higher Revenue: {revenue_result}\n"
                f"Higher Profit: {profit_result}"
            )

        else:

            return (
                "There are not enough products "
                "available for comparison."
            )
    # --------------------------------------------------------
    # STEP 57: PRODUCT COMPARISON
    # --------------------------------------------------------

    elif "compare" in q and "product" in q:

        products = data["Product"].unique()

        if len(products) >= 2:

            product_1 = products[0]
            product_2 = products[1]

            comparison = (
                data.groupby("Product")[["Revenue", "Profit"]]
                .sum()
            )

            revenue_1 = comparison.loc[product_1, "Revenue"]
            revenue_2 = comparison.loc[product_2, "Revenue"]

            profit_1 = comparison.loc[product_1, "Profit"]
            profit_2 = comparison.loc[product_2, "Profit"]

            if revenue_1 > revenue_2:
                revenue_result = product_1
            elif revenue_2 > revenue_1:
                revenue_result = product_2
            else:
                revenue_result = "Both products have equal revenue"

            if profit_1 > profit_2:
                profit_result = product_1
            elif profit_2 > profit_1:
                profit_result = product_2
            else:
                profit_result = "Both products have equal profit"

            return (
                f"Product comparison:\n\n"
                f"{product_1} Revenue: {revenue_1:.2f}\n"
                f"{product_2} Revenue: {revenue_2:.2f}\n\n"
                f"{product_1} Profit: {profit_1:.2f}\n"
                f"{product_2} Profit: {profit_2:.2f}\n\n"
                f"Higher Revenue: {revenue_result}\n"
                f"Higher Profit: {profit_result}"
            )

        else:

            return "There are not enough products for comparison."

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    else:

        return (
            "I can currently answer questions about "
            "revenue, profit, quantity, price, products, "
            "regions, and profit margins."
        )

# ============================================================
# REAL AI + LOCAL FALLBACK
# ============================================================

def ask_ai(question, context):

    try:

        prompt = f"""
You are an AI Data Analyst.

You are analyzing a business dataset.

BUSINESS DATA:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer using only the provided business data.
- Do not invent numbers or facts.
- If the exact answer cannot be determined, say so.
- Explain the answer in simple business language.
- Mention relevant numbers when available.
- If useful, give one short business recommendation.
"""

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        return response.output_text

    except Exception:

        print("\n[API unavailable - using local data analysis]")
        return ask_local_ai(question)


# ============================================================
# STEP 53: MULTI-QUESTION AI CHAT
# ============================================================

print("\n--- REAL AI DATA ANALYST ---")
print("\n--- AI DATA ANALYST CHAT ---")
print("Type 'exit' to stop the chat.")

while True:

    user_question = input("\nYou: ")

    if user_question.lower() == "exit":
        print("\nAI Data Analyst: Goodbye!")
        break

    ai_answer = ask_ai(
        user_question,
        ai_context
    )

    print("\nAI Data Analyst:")
    print(ai_answer)
