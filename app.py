import streamlit as st
import pandas as pd
import io

from report import (
    generate_dataset_summary,
    generate_data_quality_report,
    generate_business_report,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 🎨 PROFESSIONAL SIDEBAR
# =========================================================

st.sidebar.title("📊 AI Data Analyst")

st.sidebar.markdown(
    "### Dashboard Navigation"
)

dashboard_section = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📊 Analytics",
        "🔎 Anomalies",
        "🔮 Forecast",
        "🤖 Ask Your Data",
        "📄 Reports",
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 📌 Project Features")

st.sidebar.write("✅ Automated EDA")
st.sidebar.write("✅ Business Insights")
st.sidebar.write("✅ Anomaly Detection")
st.sidebar.write("✅ Revenue Forecasting")
st.sidebar.write("✅ Natural Language Q&A")
st.sidebar.write("✅ Report Generation")
st.sidebar.write("✅ CSV / Excel Export")

st.sidebar.divider()

st.sidebar.caption(
    "AI Data Analyst\n"
    "B.Tech AI/ML Project"
)
st.sidebar.divider()
st.sidebar.caption("AI Data Analyst • v1.0")


# =========================================================
# TITLE
# =========================================================

# =========================================================
# 🧭 DASHBOARD NAVIGATION
# =========================================================
st.markdown("### 🧭 Dashboard Section")

section_info = {
    "🏠 Overview": "Business overview and key performance metrics",
    "📊 Analytics": "Visual analytics and business performance",
    "🔎 Anomalies": "Outlier and anomaly detection",
    "🔮 Forecast": "Revenue forecasting",
    "🤖 Ask Your Data": "Ask questions about your dataset",
    "📄 Reports": "Reports and export options",
}

st.info(
    f"📌 **{dashboard_section}** — {section_info[dashboard_section]}"
)

st.title("📊 AI Data Analyst")
st.write(
    "Upload your dataset and explore automated analysis, "
    "business insights, visual analytics and data quality."
)

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"],
)


# =========================================================
# NO FILE UPLOADED
# =========================================================

if uploaded_file is None:
    st.info("👆 Upload a CSV or Excel file to start analyzing.")
    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:
    if uploaded_file.name.lower().endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()


if data.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()


# =========================================================
# REQUIRED RAW COLUMNS
# =========================================================

required_columns = {"Quantity", "Price", "Cost"}
missing_required = required_columns - set(data.columns)

if missing_required:
    st.error(
        "This dataset is missing required columns: "
        + ", ".join(sorted(missing_required))
    )
    st.info(
        "For this project, the dataset should contain at least "
        "Quantity, Price and Cost. Product, Region and Date enable "
        "additional analytics."
    )
    st.stop()


# =========================================================
# CREATE REVENUE AND PROFIT
# =========================================================

if "Revenue" not in data.columns:
    data["Revenue"] = data["Quantity"] * data["Price"]

if "Profit" not in data.columns:
    data["Profit"] = data["Revenue"] - (
        data["Quantity"] * data["Cost"]
    )


st.success(
    f"Dataset loaded successfully: {uploaded_file.name}"
)


# =========================================================
# 🧩 SHARED ANALYSIS HELPERS
# =========================================================
def calculate_quality_metrics(df):
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    total_cells = df.shape[0] * df.shape[1]
    missing_percentage = (missing_count / total_cells * 100) if total_cells else 0
    quality_score = 100
    if missing_percentage > 0:
        quality_score -= min(50, missing_percentage)
    if duplicate_count > 0 and len(df) > 0:
        duplicate_percentage = duplicate_count / len(df) * 100
        quality_score -= min(30, duplicate_percentage)
    return round(max(0, quality_score), 1), missing_count, duplicate_count

def get_revenue_outliers(df):
    q1 = df["Revenue"].quantile(0.25)
    q3 = df["Revenue"].quantile(0.75)
    iqr = q3 - q1
    return df[(df["Revenue"] < q1 - 1.5 * iqr) | (df["Revenue"] > q3 + 1.5 * iqr)]

def get_profit_outliers(df):
    q1 = df["Profit"].quantile(0.25)
    q3 = df["Profit"].quantile(0.75)
    iqr = q3 - q1
    return df[(df["Profit"] < q1 - 1.5 * iqr) | (df["Profit"] > q3 + 1.5 * iqr)]

quality_score, missing_count, duplicate_count = calculate_quality_metrics(data)
revenue_outliers = get_revenue_outliers(data)
profit_outliers = get_profit_outliers(data)


# =========================================================
# 🏠 OVERVIEW
# =========================================================
def render_overview():

    # =========================================================
    # DATASET PREVIEW
    # =========================================================

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True,
    )


    # =========================================================
    # BUSINESS METRICS
    # =========================================================

    st.subheader("💰 Business Performance")

    total_revenue = data["Revenue"].sum()
    total_profit = data["Profit"].sum()
    total_quantity = data["Quantity"].sum()

    if total_revenue != 0:
        profit_margin = (total_profit / total_revenue) * 100
    else:
        profit_margin = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Revenue", f"₹{total_revenue:,.2f}")

    with col2:
        st.metric("Total Profit", f"₹{total_profit:,.2f}")

    with col3:
        st.metric("Total Quantity", f"{total_quantity:,}")

    with col4:
        st.metric("Profit Margin", f"{profit_margin:.2f}%")


    # =========================================================
    # AUTOMATED REPORTS
    # =========================================================

    st.subheader("🤖 Automated Analysis")

    report_col1, report_col2, report_col3 = st.columns(3)

    with report_col1:
        if st.button("📋 Dataset Summary", use_container_width=True):
            st.text(generate_dataset_summary(data))

    with report_col2:
        if st.button("🔍 Data Quality Report", use_container_width=True):
            st.text(generate_data_quality_report(data))

    with report_col3:
        if st.button("📊 Business Report", use_container_width=True):
            st.text(generate_business_report(data))
    # =========================================================
    # COLUMN INFORMATION
    # =========================================================

    st.subheader("🧾 Column Information")

    column_info = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Missing Values": data.isnull().sum().values,
    })

    st.dataframe(
        column_info,
        use_container_width=True,
    )
    # =========================================================
    # 🎯 DATA QUALITY SCORE
    # =========================================================

    st.subheader("🎯 Data Quality Score")

    missing_count = data.isnull().sum().sum()
    duplicate_count = data.duplicated().sum()
    total_cells = data.shape[0] * data.shape[1]

    if total_cells > 0:
        missing_percentage = (
            missing_count / total_cells
        ) * 100
    else:
        missing_percentage = 0

    quality_score = 100

    if missing_percentage > 0:
        quality_score -= min(
            50,
            missing_percentage
        )

    if duplicate_count > 0:
        duplicate_percentage = (
            duplicate_count / len(data)
        ) * 100

        quality_score -= min(
            30,
            duplicate_percentage
        )

    quality_score = max(
        0,
        round(quality_score, 1)
    )

    quality_col1, quality_col2, quality_col3 = st.columns(3)

    with quality_col1:
        st.metric(
            "Quality Score",
            f"{quality_score}/100"
        )

    with quality_col2:
        st.metric(
            "Missing Values",
            f"{missing_count:,}"
        )

    with quality_col3:
        st.metric(
            "Duplicate Rows",
            f"{duplicate_count:,}"
        )

    if quality_score >= 90:

        st.success(
            "🟢 Dataset quality is excellent."
        )

    elif quality_score >= 70:

        st.warning(
            "🟡 Dataset quality is acceptable but "
            "some cleaning may be required."
        )

    else:

        st.error(
            "🔴 Dataset requires significant data cleaning."
        )


    # =========================================================
    # 🏆 TOP PERFORMERS
    # =========================================================

    st.subheader("🏆 Top Business Performers")

    top_col1, top_col2, top_col3 = st.columns(3)

    with top_col1:

        if "Product" in data.columns:

            top_product_data = (
                data.groupby("Product")["Revenue"]
                .sum()
                .sort_values(ascending=False)
            )

            if not top_product_data.empty:

                st.metric(
                    "🏆 Top Product",
                    str(top_product_data.idxmax())
                )

                st.caption(
                    f"Revenue: ₹{top_product_data.max():,.2f}"
                )

    with top_col2:

        if "Region" in data.columns:

            top_region_data = (
                data.groupby("Region")["Revenue"]
                .sum()
                .sort_values(ascending=False)
            )

            if not top_region_data.empty:

                st.metric(
                    "🌍 Top Region",
                    str(top_region_data.idxmax())
                )

                st.caption(
                    f"Revenue: ₹{top_region_data.max():,.2f}"
                )

    with top_col3:

        if "Product" in data.columns:

            profit_product_data = (
                data.groupby("Product")["Profit"]
                .sum()
                .sort_values(ascending=False)
            )

            if not profit_product_data.empty:

                st.metric(
                    "💰 Highest Profit Product",
                    str(profit_product_data.idxmax())
                )

                st.caption(
                    f"Profit: ₹{profit_product_data.max():,.2f}"
                )


    # =========================================================
    # 📅 DATE RANGE ANALYSIS
    # =========================================================

    if "Date" in data.columns:

        st.subheader("📅 Date Range Analysis")

        date_analysis = data.copy()

        date_analysis["Date"] = pd.to_datetime(
            date_analysis["Date"],
            errors="coerce"
        )

        valid_date_analysis = date_analysis.dropna(
            subset=["Date"]
        )

        if not valid_date_analysis.empty:

            min_date = valid_date_analysis["Date"].min()
            max_date = valid_date_analysis["Date"].max()

            date_col1, date_col2, date_col3 = st.columns(3)

            with date_col1:

                st.metric(
                    "Start Date",
                    min_date.strftime("%d %b %Y")
                )

            with date_col2:

                st.metric(
                    "End Date",
                    max_date.strftime("%d %b %Y")
                )

            with date_col3:

                total_days = (
                    max_date - min_date
                ).days + 1

                st.metric(
                    "Analysis Period",
                    f"{total_days} days"
                )
    # =========================================================
    # 🚨 BUSINESS ALERTS
    # =========================================================

    st.subheader("🚨 Business Alerts")

    alerts = []

    if total_profit < 0:

        alerts.append(
            "🔴 Overall business profit is negative."
        )

    elif total_profit == 0:

        alerts.append(
            "🟡 Overall business profit is zero."
        )

    else:

        alerts.append(
            "🟢 Overall business performance is profitable."
        )

    if "revenue_outliers" in locals():

        if len(revenue_outliers) > 0:

            alerts.append(
                f"🟠 {len(revenue_outliers)} revenue "
                "outlier transaction(s) detected."
            )

    if "profit_outliers" in locals():

        if len(profit_outliers) > 0:

            alerts.append(
                f"🟠 {len(profit_outliers)} profit "
                "outlier transaction(s) detected."
            )

    if missing_count > 0:

        alerts.append(
            f"🟡 Dataset contains {missing_count} "
            "missing value(s)."
        )

    if duplicate_count > 0:

        alerts.append(
            f"🟡 Dataset contains {duplicate_count} "
            "duplicate row(s)."
        )

    if len(alerts) == 0:

        st.success(
            "No major business alerts detected."
        )

    else:

        for alert in alerts:

            st.info(alert)


    st.divider()
    st.subheader("📌 Executive Summary")
    exec1, exec2, exec3, exec4 = st.columns(4)
    with exec1:
        st.metric("💰 Revenue", f"₹{total_revenue:,.0f}")
    with exec2:
        st.metric("📈 Profit", f"₹{total_profit:,.0f}")
    with exec3:
        st.metric("📦 Quantity", f"{total_quantity:,.0f}")
    with exec4:
        st.metric("🎯 Quality", f"{quality_score}/100")


# =========================================================
# 📊 ANALYTICS
# =========================================================
def render_analytics():

    st.subheader("📊 Visual Analytics")

    if "Product" in data.columns:
        st.markdown("### 🛍️ Product Performance")
        product_analysis = (
            data.groupby("Product")[["Revenue", "Profit"]]
            .sum()
            .sort_values("Revenue", ascending=False)
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Revenue by Product**")
            st.bar_chart(product_analysis["Revenue"])
        with col2:
            st.markdown("**Profit by Product**")
            st.bar_chart(product_analysis["Profit"])
        st.dataframe(product_analysis, use_container_width=True)
    else:
        st.info("Product analytics require a Product column.")

    if "Region" in data.columns:
        st.markdown("### 🌍 Region Performance")
        region_analysis = (
            data.groupby("Region")[["Revenue", "Profit"]]
            .sum()
            .sort_values("Revenue", ascending=False)
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Revenue by Region**")
            st.bar_chart(region_analysis["Revenue"])
        with col2:
            st.markdown("**Profit by Region**")
            st.bar_chart(region_analysis["Profit"])
        st.dataframe(region_analysis, use_container_width=True)
    else:
        st.info("Region analytics require a Region column.")

    st.markdown("### 📅 Monthly Revenue Trend")
    if "Date" in data.columns:
        monthly_data = data.copy()
        monthly_data["Date"] = pd.to_datetime(monthly_data["Date"], errors="coerce")
        valid_monthly = monthly_data.dropna(subset=["Date"])
        monthly_revenue = (
            valid_monthly.groupby(valid_monthly["Date"].dt.to_period("M"))["Revenue"]
            .sum()
        )
        monthly_revenue.index = monthly_revenue.index.astype(str)
        if not monthly_revenue.empty:
            st.line_chart(monthly_revenue)
        else:
            st.info("No valid dates were found for monthly analysis.")
    else:
        st.info("Monthly revenue trend requires a Date column.")

    st.subheader("🔬 Advanced Analytics")

    if "Product" in data.columns:
        st.markdown("### 💰 Profit Margin by Product")
        product_margin = data.groupby("Product")[["Revenue", "Profit"]].sum()
        product_margin["Profit Margin"] = (
            product_margin["Profit"] /
            product_margin["Revenue"].replace(0, pd.NA)
        ) * 100
        product_margin["Profit Margin"] = product_margin["Profit Margin"].fillna(0)
        st.bar_chart(product_margin["Profit Margin"].sort_values(ascending=False))

    if "Region" in data.columns:
        st.markdown("### 🌍 Profit Margin by Region")
        region_margin = data.groupby("Region")[["Revenue", "Profit"]].sum()
        region_margin["Profit Margin"] = (
            region_margin["Profit"] /
            region_margin["Revenue"].replace(0, pd.NA)
        ) * 100
        region_margin["Profit Margin"] = region_margin["Profit Margin"].fillna(0)
        st.bar_chart(region_margin["Profit Margin"].sort_values(ascending=False))

    if "Product" in data.columns:
        st.markdown("### 📦 Quantity Sold by Product")
        quantity_by_product = (
            data.groupby("Product")["Quantity"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(quantity_by_product)

    if "Product" in data.columns:
        st.markdown("### 📈 Revenue vs Profit")
        comparison_data = data.groupby("Product")[["Revenue", "Profit"]].sum()
        st.dataframe(comparison_data, use_container_width=True)

    st.markdown("### 🔗 Correlation Analysis")
    numeric_data = data.select_dtypes(include="number")
    if len(numeric_data.columns) >= 2:
        st.dataframe(numeric_data.corr().round(2), use_container_width=True)
    else:
        st.info("Not enough numeric columns available for correlation analysis.")

    if "Category" in data.columns:
        st.subheader("📦 Category Analysis")
        category_analysis = (
            data.groupby("Category")[["Revenue", "Profit", "Quantity"]]
            .sum()
            .sort_values("Revenue", ascending=False)
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Revenue by Category**")
            st.bar_chart(category_analysis["Revenue"])
        with col2:
            st.markdown("**Profit by Category**")
            st.bar_chart(category_analysis["Profit"])
        st.dataframe(category_analysis, use_container_width=True)

    if "Product" in data.columns:
        st.subheader("📈 Revenue Contribution by Product")
        contribution_data = data.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
        total_product_revenue = contribution_data.sum()
        if total_product_revenue != 0:
            contribution_table = pd.DataFrame({
                "Revenue": contribution_data,
                "Contribution %": (contribution_data / total_product_revenue * 100).round(2),
            })
            st.dataframe(contribution_table, use_container_width=True)

    st.divider()
    st.subheader("🎛️ Interactive Dataset Filter")
    filtered_data = data.copy()
    col1, col2 = st.columns(2)

    with col1:
        if "Product" in data.columns:
            selected_products = st.multiselect(
                "Select Product",
                sorted(data["Product"].dropna().unique()),
                key="analytics_products",
            )
            if selected_products:
                filtered_data = filtered_data[filtered_data["Product"].isin(selected_products)]

    with col2:
        if "Region" in data.columns:
            selected_regions = st.multiselect(
                "Select Region",
                sorted(data["Region"].dropna().unique()),
                key="analytics_regions",
            )
            if selected_regions:
                filtered_data = filtered_data[filtered_data["Region"].isin(selected_regions)]

    st.write(f"Showing **{len(filtered_data)}** of **{len(data)}** records.")
    st.dataframe(filtered_data, use_container_width=True)

    csv_data = filtered_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Dataset",
        csv_data,
        "filtered_dataset.csv",
        "text/csv",
        key="analytics_csv_download",
    )


# =========================================================
# 🔎 ANOMALIES
# =========================================================
def render_anomalies():

    # =====================================================
    # 🔎 OUTLIER & ANOMALY ANALYSIS
    # =====================================================

    st.subheader("🔎 Outlier & Anomaly Analysis")

    # -----------------------------------------------------
    # REVENUE OUTLIERS
    # -----------------------------------------------------

    st.markdown("### 💰 Revenue Outliers")

    revenue_q1 = data["Revenue"].quantile(0.25)
    revenue_q3 = data["Revenue"].quantile(0.75)

    revenue_iqr = revenue_q3 - revenue_q1

    revenue_lower = revenue_q1 - (1.5 * revenue_iqr)
    revenue_upper = revenue_q3 + (1.5 * revenue_iqr)

    revenue_outliers = data[
        (data["Revenue"] < revenue_lower) |
        (data["Revenue"] > revenue_upper)
    ]

    outlier_col1, outlier_col2 = st.columns(2)

    with outlier_col1:
        st.metric(
            "Revenue Outliers",
            len(revenue_outliers)
        )

    with outlier_col2:
        st.metric(
            "Normal Revenue Records",
            len(data) - len(revenue_outliers)
        )

    if len(revenue_outliers) > 0:

        st.dataframe(
            revenue_outliers[
                [
                    col for col in
                    ["Date", "Product", "Region", "Revenue", "Profit"]
                    if col in data.columns
                ]
            ],
            use_container_width=True
        )

    else:

        st.success(
            "No unusual revenue transactions detected."
        )

    # -----------------------------------------------------
    # PROFIT OUTLIERS
    # -----------------------------------------------------

    st.markdown("### 📈 Profit Outliers")

    profit_q1 = data["Profit"].quantile(0.25)
    profit_q3 = data["Profit"].quantile(0.75)

    profit_iqr = profit_q3 - profit_q1

    profit_lower = profit_q1 - (1.5 * profit_iqr)
    profit_upper = profit_q3 + (1.5 * profit_iqr)

    profit_outliers = data[
        (data["Profit"] < profit_lower) |
        (data["Profit"] > profit_upper)
    ]

    profit_col1, profit_col2 = st.columns(2)

    with profit_col1:
        st.metric(
            "Profit Outliers",
            len(profit_outliers)
        )

    with profit_col2:
        st.metric(
            "Normal Profit Records",
            len(data) - len(profit_outliers)
        )

    if len(profit_outliers) > 0:

        st.dataframe(
            profit_outliers[
                [
                    col for col in
                    ["Date", "Product", "Region", "Revenue", "Profit"]
                    if col in data.columns
                ]
            ],
            use_container_width=True
        )

    else:

        st.success(
            "No unusual profit transactions detected."
        )

    # -----------------------------------------------------
    # HIGH VALUE TRANSACTIONS
    # -----------------------------------------------------

    st.markdown("### 💎 High Value Transactions")

    high_value_threshold = data["Revenue"].quantile(0.90)

    high_value_transactions = data[
        data["Revenue"] >= high_value_threshold
    ]

    st.write(
        f"Transactions above the 90th percentile "
        f"of revenue: {len(high_value_transactions)}"
    )

    if len(high_value_transactions) > 0:

        st.dataframe(
            high_value_transactions[
                [
                    col for col in
                    ["Date", "Product", "Region", "Quantity", "Revenue", "Profit"]
                    if col in data.columns
                ]
            ].sort_values(
                "Revenue",
                ascending=False
            ),
            use_container_width=True
        )
    # =====================================================
    # 💡 AUTOMATED BUSINESS INSIGHTS
    # =====================================================

    st.subheader("💡 Automated Business Insights")

    insights = []

    # -----------------------------------------------------
    # OVERALL PERFORMANCE
    # -----------------------------------------------------

    total_revenue = data["Revenue"].sum()
    total_profit = data["Profit"].sum()

    if total_revenue != 0:
        overall_margin = (total_profit / total_revenue) * 100
    else:
        overall_margin = 0

    insights.append(
        f"Total revenue generated is ₹{total_revenue:,.2f} "
        f"with total profit of ₹{total_profit:,.2f}."
    )

    insights.append(
        f"Overall profit margin is {overall_margin:.2f}%."
    )

    # -----------------------------------------------------
    # TOP PRODUCT
    # -----------------------------------------------------

    if "Product" in data.columns:

        product_revenue = (
            data.groupby("Product")["Revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        top_product = product_revenue.idxmax()
        top_product_revenue = product_revenue.max()

        insights.append(
            f"{top_product} is the highest revenue-generating "
            f"product with ₹{top_product_revenue:,.2f} revenue."
        )

        product_profit = (
            data.groupby("Product")["Profit"]
            .sum()
        )

        top_profit_product = product_profit.idxmax()
        top_profit_value = product_profit.max()

        insights.append(
            f"{top_profit_product} generated the highest total "
            f"profit of ₹{top_profit_value:,.2f}."
        )

    # -----------------------------------------------------
    # TOP REGION
    # -----------------------------------------------------

    if "Region" in data.columns:

        region_revenue = (
            data.groupby("Region")["Revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        top_region = region_revenue.idxmax()
        top_region_revenue = region_revenue.max()

        insights.append(
            f"{top_region} is the highest revenue-generating "
            f"region with ₹{top_region_revenue:,.2f} revenue."
        )

        region_profit = (
            data.groupby("Region")["Profit"]
            .sum()
        )

        top_profit_region = region_profit.idxmax()
        top_region_profit = region_profit.max()

        insights.append(
            f"{top_profit_region} generated the highest total "
            f"profit of ₹{top_region_profit:,.2f}."
        )

    # -----------------------------------------------------
    # MONTHLY PERFORMANCE
    # -----------------------------------------------------

    if "Date" in data.columns:

        insight_dates = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        valid_dates = data.copy()
        valid_dates["Insight_Date"] = insight_dates

        valid_dates = valid_dates.dropna(
            subset=["Insight_Date"]
        )

        if len(valid_dates) > 0:

            monthly_insight = (
                valid_dates
                .groupby(
                    valid_dates["Insight_Date"].dt.to_period("M")
                )["Revenue"]
                .sum()
            )

            if len(monthly_insight) > 0:

                best_month = monthly_insight.idxmax()
                best_month_revenue = monthly_insight.max()

                insights.append(
                    f"{best_month} was the highest revenue month "
                    f"with ₹{best_month_revenue:,.2f} revenue."
                )

    # -----------------------------------------------------
    # OUTLIER INSIGHTS
    # -----------------------------------------------------

    if "revenue_outliers" in locals():

        if len(revenue_outliers) > 0:

            insights.append(
                f"{len(revenue_outliers)} unusual revenue "
                f"transactions were detected."
            )

        else:

            insights.append(
                "No unusual revenue transactions were detected."
            )

    if "profit_outliers" in locals():

        if len(profit_outliers) > 0:

            insights.append(
                f"{len(profit_outliers)} unusual profit "
                f"transactions were detected."
            )

        else:

            insights.append(
                "No unusual profit transactions were detected."
            )

    # -----------------------------------------------------
    # DISPLAY INSIGHTS
    # -----------------------------------------------------

    for number, insight in enumerate(insights, start=1):

        st.info(
            f"**Insight {number}:** {insight}"
        )


    # =====================================================
    # 📋 DATASET FILTER
    # =====================================================

    st.subheader("🎛️ Dataset Filter")

    filter_col1, filter_col2 = st.columns(2)

    filtered_data = data.copy()

    with filter_col1:

        if "Product" in data.columns:

            selected_products = st.multiselect(
                "Select Product",
                options=sorted(
                    data["Product"].dropna().unique()
                )
            )

            if selected_products:

                filtered_data = filtered_data[
                    filtered_data["Product"].isin(
                        selected_products
                    )
                ]

    with filter_col2:

        if "Region" in data.columns:

            selected_regions = st.multiselect(
                "Select Region",
                options=sorted(
                    data["Region"].dropna().unique()
                )
            )

            if selected_regions:

                filtered_data = filtered_data[
                    filtered_data["Region"].isin(
                        selected_regions
                    )
                ]

    st.write(
        f"Showing **{len(filtered_data)}** "
        f"of **{len(data)}** records."
    )

    st.dataframe(
        filtered_data,
        use_container_width=True
    )


    # =====================================================
    # 📥 DOWNLOAD FILTERED DATA
    # =====================================================

    st.subheader("📥 Export Data")

    csv_data = filtered_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=csv_data,
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )


    # =====================================================
    # 📊 FINAL DASHBOARD SUMMARY
    # =====================================================

    st.subheader("📊 Dashboard Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:

        st.metric(
            "Records",
            len(filtered_data)
        )

    with summary_col2:

        st.metric(
            "Revenue",
            f"₹{filtered_data['Revenue'].sum():,.0f}"
        )

    with summary_col3:

        st.metric(
            "Profit",
            f"₹{filtered_data['Profit'].sum():,.0f}"
        )

    with summary_col4:

        filtered_revenue = filtered_data["Revenue"].sum()
        filtered_profit = filtered_data["Profit"].sum()

        if filtered_revenue != 0:
            filtered_margin = (
                filtered_profit /
                filtered_revenue
            ) * 100
        else:
            filtered_margin = 0

        st.metric(
            "Profit Margin",
            f"{filtered_margin:.2f}%"
        )

# =========================================================
# 🔮 FORECAST
# =========================================================
def render_forecast():

    # =========================================================
    # 🔮 REVENUE FORECASTING
    # =========================================================

    st.subheader("🔮 Revenue Forecasting")

    if "Date" in data.columns:

        forecast_data = data.copy()

        forecast_data["Date"] = pd.to_datetime(
            forecast_data["Date"],
            errors="coerce"
        )

        forecast_data = forecast_data.dropna(
            subset=["Date"]
        )

        if not forecast_data.empty:

            monthly_forecast = (
                forecast_data
                .groupby(
                    forecast_data["Date"].dt.to_period("M")
                )["Revenue"]
                .sum()
                .sort_index()
            )

            if len(monthly_forecast) >= 2:

                st.markdown("### 📈 Historical Revenue")

                forecast_chart_data = monthly_forecast.copy()
                forecast_chart_data.index = (
                    forecast_chart_data.index.astype(str)
                )

                st.line_chart(
                    forecast_chart_data
                )

                # -------------------------------------------------
                # SIMPLE LINEAR FORECAST
                # -------------------------------------------------

                x = list(range(len(monthly_forecast)))
                y = monthly_forecast.values

                x_mean = sum(x) / len(x)
                y_mean = sum(y) / len(y)

                numerator = sum(
                    (x[i] - x_mean) *
                    (y[i] - y_mean)
                    for i in range(len(x))
                )

                denominator = sum(
                    (x[i] - x_mean) ** 2
                    for i in range(len(x))
                )

                if denominator != 0:

                    slope = numerator / denominator
                    intercept = y_mean - (slope * x_mean)

                    # Forecast next 3 months
                    future_x = [
                        len(monthly_forecast),
                        len(monthly_forecast) + 1,
                        len(monthly_forecast) + 2
                    ]

                    future_revenue = [
                        max(
                            0,
                            intercept + slope * value
                        )
                        for value in future_x
                    ]

                    last_period = monthly_forecast.index[-1]

                    future_periods = [
                        last_period + 1,
                        last_period + 2,
                        last_period + 3
                    ]

                    forecast_table = pd.DataFrame({
                        "Month": [
                            str(period)
                            for period in future_periods
                        ],
                        "Forecast Revenue": future_revenue
                    })

                    st.markdown("### 🔮 Next 3 Months Forecast")

                    st.dataframe(
                        forecast_table,
                        use_container_width=True
                    )

                    st.line_chart(
                        forecast_table.set_index("Month")
                    )

                    average_forecast = (
                        sum(future_revenue)
                        / len(future_revenue)
                    )

                    forecast_col1, forecast_col2 = st.columns(2)

                    with forecast_col1:

                        st.metric(
                            "Average Forecast Revenue",
                            f"₹{average_forecast:,.2f}"
                        )

                    with forecast_col2:

                        st.metric(
                            "Next Month Forecast",
                            f"₹{future_revenue[0]:,.2f}"
                        )

                    if future_revenue[0] > monthly_forecast.iloc[-1]:

                        st.success(
                            "📈 Forecast indicates increasing revenue "
                            "compared with the latest month."
                        )

                    elif future_revenue[0] < monthly_forecast.iloc[-1]:

                        st.warning(
                            "📉 Forecast indicates decreasing revenue "
                            "compared with the latest month."
                        )

                    else:

                        st.info(
                            "➡️ Forecast indicates relatively stable "
                            "revenue."
                        )

                else:

                    st.info(
                        "Revenue values do not show enough variation "
                        "for forecasting."
                    )

            else:

                st.info(
                    "At least 2 months of data are required "
                    "for revenue forecasting."
                )

        else:

            st.info(
                "No valid dates available for forecasting."
            )

    else:

        st.info(
            "Revenue forecasting requires a Date column."
        )


# =========================================================
# 🤖 ASK YOUR DATA
# =========================================================
def render_ask_data():

    st.subheader("🤖 Ask Your Data")
    st.write(
        "Ask questions about revenue, profit, quantity, products, regions or dataset information."
    )

    st.markdown("### 💡 Try asking")
    st.caption(
        "Which product generated the highest revenue? • "
        "Which region generated the highest revenue? • "
        "What is the total profit? • "
        "What is the profit margin?"
    )

    user_question = st.text_input(
        "Ask a question",
        placeholder="Example: Which product generated the highest revenue?",
        key="ask_data_question",
    )

    if not user_question:
        return

    question = user_question.lower().strip()

    if (
        "total revenue" in question
        or "revenue generated" in question
        or "overall revenue" in question
        or "how much revenue" in question
    ):
        st.success(f"💰 Total Revenue: ₹{data['Revenue'].sum():,.2f}")

    elif (
        "total profit" in question
        or "overall profit" in question
        or "profit generated" in question
        or "how much profit" in question
    ):
        st.success(f"💰 Total Profit: ₹{data['Profit'].sum():,.2f}")

    elif "average revenue" in question:
        st.success(f"📊 Average Transaction Revenue: ₹{data['Revenue'].mean():,.2f}")

    elif "average profit" in question:
        st.success(f"📊 Average Transaction Profit: ₹{data['Profit'].mean():,.2f}")

    elif "profit margin" in question:
        revenue = data["Revenue"].sum()
        profit = data["Profit"].sum()
        margin = (profit / revenue * 100) if revenue != 0 else 0
        st.success(f"📊 Overall Profit Margin: {margin:.2f}%")

    elif (
        "top product" in question
        or "best product" in question
        or "highest revenue product" in question
        or ("product" in question and "highest revenue" in question)
        or ("which product" in question and "revenue" in question)
        or ("product" in question and "most revenue" in question)
    ):
        if "Product" in data.columns:
            product_revenue = data.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
            st.success(
                f"🏆 Top Product: {product_revenue.idxmax()} "
                f"(₹{product_revenue.max():,.2f} revenue)"
            )
        else:
            st.info("This dataset does not contain a Product column.")

    elif (
        "top region" in question
        or "best region" in question
        or "highest revenue region" in question
        or ("region" in question and "highest revenue" in question)
        or ("which region" in question and "revenue" in question)
        or ("region" in question and "most revenue" in question)
    ):
        if "Region" in data.columns:
            region_revenue = data.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
            st.success(
                f"🌍 Top Region: {region_revenue.idxmax()} "
                f"(₹{region_revenue.max():,.2f} revenue)"
            )
        else:
            st.info("This dataset does not contain a Region column.")

    elif "total quantity" in question or "quantity sold" in question:
        st.success(f"📦 Total Quantity Sold: {data['Quantity'].sum():,.0f}")

    elif "how many rows" in question or "number of rows" in question or "total rows" in question:
        st.success(f"📋 Total Rows: {len(data):,}")

    elif "how many columns" in question or "number of columns" in question or "total columns" in question:
        st.success(f"📋 Total Columns: {len(data.columns):,}")

    elif "missing values" in question or "missing data" in question:
        st.success(f"🔍 Total Missing Values: {int(data.isnull().sum().sum()):,}")

    elif "duplicate" in question or "duplicates" in question:
        st.success(f"🔍 Duplicate Rows: {int(data.duplicated().sum()):,}")

    elif "columns" in question or "column names" in question:
        st.write("📋 Dataset Columns:")
        st.write(", ".join(data.columns))

    elif "dataset" in question or "data information" in question:
        st.write(f"📊 Rows: {len(data):,}")
        st.write(f"📋 Columns: {len(data.columns):,}")
        st.write(f"💰 Revenue: ₹{data['Revenue'].sum():,.2f}")
        st.write(f"💰 Profit: ₹{data['Profit'].sum():,.2f}")

    else:
        st.info(
            "I can currently answer questions about revenue, profit, quantity, "
            "profit margin, products, regions, rows, columns, missing values "
            "and duplicates."
        )


# =========================================================
# 📄 REPORTS
# =========================================================
def render_reports():

    st.subheader("📄 Reports & Exports")
    st.write("Generate a business report or export the complete analysis to Excel.")

    total_revenue = data["Revenue"].sum()
    total_profit = data["Profit"].sum()
    total_quantity = data["Quantity"].sum()
    profit_margin = (total_profit / total_revenue * 100) if total_revenue != 0 else 0

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Revenue", f"₹{total_revenue:,.0f}")
    with metric_col2:
        st.metric("Profit", f"₹{total_profit:,.0f}")
    with metric_col3:
        st.metric("Quantity", f"{total_quantity:,.0f}")
    with metric_col4:
        st.metric("Profit Margin", f"{profit_margin:.2f}%")

    st.subheader("📥 Download Business Report")

    download_report = generate_business_report(data)
    download_report += "\n\n================ DASHBOARD METRICS ================\n\n"
    download_report += f"Total Revenue: ₹{total_revenue:,.2f}\n"
    download_report += f"Total Profit: ₹{total_profit:,.2f}\n"
    download_report += f"Total Quantity: {total_quantity:,.0f}\n"
    download_report += f"Profit Margin: {profit_margin:.2f}%\n"

    if "Product" in data.columns:
        report_product = data.groupby("Product")["Revenue"].sum().idxmax()
        download_report += f"Top Product: {report_product}\n"

    if "Region" in data.columns:
        report_region = data.groupby("Region")["Revenue"].sum().idxmax()
        download_report += f"Top Region: {report_region}\n"

    download_report += "\n====================================================\n"

    st.download_button(
        label="📥 Download Business Report",
        data=download_report,
        file_name="AI_Data_Analyst_Report.txt",
        mime="text/plain",
        key="business_report_download",
    )

    st.subheader("📊 Download Complete Excel Analysis")
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Dataset", index=False)

        summary_export = pd.DataFrame({
            "Metric": [
                "Total Revenue", "Total Profit", "Total Quantity", "Profit Margin",
                "Rows", "Columns", "Missing Values", "Duplicate Rows"
            ],
            "Value": [
                total_revenue, total_profit, total_quantity, profit_margin,
                len(data), len(data.columns), missing_count, duplicate_count
            ],
        })
        summary_export.to_excel(writer, sheet_name="Summary", index=False)

        if "Product" in data.columns:
            product_export = (
                data.groupby("Product")[["Revenue", "Profit", "Quantity"]]
                .sum()
                .sort_values("Revenue", ascending=False)
            )
            product_export.to_excel(writer, sheet_name="Product Analysis")

        if "Region" in data.columns:
            region_export = (
                data.groupby("Region")[["Revenue", "Profit", "Quantity"]]
                .sum()
                .sort_values("Revenue", ascending=False)
            )
            region_export.to_excel(writer, sheet_name="Region Analysis")

    excel_buffer.seek(0)
    st.download_button(
        label="📊 Download Complete Excel Analysis",
        data=excel_buffer,
        file_name="AI_Data_Analyst_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="excel_analysis_download",
    )

    st.subheader("🧠 Suggested Questions")
    suggested_questions = [
        "What is the total revenue?",
        "What is the total profit?",
        "Which product generated the highest revenue?",
        "Which region generated the highest revenue?",
        "What is the overall profit margin?",
        "How many rows are in the dataset?",
        "How many missing values are there?",
        "How many duplicate rows are there?",
    ]
    for suggestion in suggested_questions:
        st.markdown(f"• {suggestion}")

    st.divider()
    st.subheader("🚀 Project Completion")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("✅ Automated EDA")
    with c2:
        st.success("✅ Business Analytics")
    with c3:
        st.success("✅ Forecasting")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.success("✅ Anomaly Detection")
    with c5:
        st.success("✅ Ask Your Data")
    with c6:
        st.success("✅ Report Export")

    st.caption(
        "AI Data Analyst • Automated data understanding, analytics, insights, forecasting and reporting"
    )


# =========================================================
# 🚦 SECTION ROUTER
# =========================================================
if dashboard_section == "🏠 Overview":
    render_overview()
elif dashboard_section == "📊 Analytics":
    render_analytics()
elif dashboard_section == "🔎 Anomalies":
    render_anomalies()
elif dashboard_section == "🔮 Forecast":
    render_forecast()
elif dashboard_section == "🤖 Ask Your Data":
    render_ask_data()
elif dashboard_section == "📄 Reports":
    render_reports()
