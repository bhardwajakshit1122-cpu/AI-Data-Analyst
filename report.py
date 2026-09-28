# ============================================================
# STEP 58: AUTOMATED DATASET SUMMARY
# ============================================================

def generate_dataset_summary(data):

    rows = len(data)
    columns = len(data.columns)

    total_revenue_value = data["Revenue"].sum()
    total_profit_value = data["Profit"].sum()

    top_product_data = (
        data.groupby("Product")["Revenue"]
        .sum()
    )

    top_product_name = top_product_data.idxmax()
    top_product_value = top_product_data.max()

    top_region_data = (
        data.groupby("Region")["Revenue"]
        .sum()
    )

    top_region_name = top_region_data.idxmax()
    top_region_value = top_region_data.max()

    summary = f"""
================ DATASET SUMMARY ================

Dataset Size:
Rows: {rows}
Columns: {columns}

Financial Performance:
Total Revenue: {total_revenue_value:.2f}
Total Profit: {total_profit_value:.2f}

Top Product:
{top_product_name}
Revenue: {top_product_value:.2f}

Top Region:
{top_region_name}
Revenue: {top_region_value:.2f}

==================================================
"""

    return summary


# ============================================================
# STEP 59: AUTOMATED DATA QUALITY REPORT
# ============================================================

def generate_data_quality_report(data):

    missing_values = data.isnull().sum().sum()
    duplicate_rows = data.duplicated().sum()

    quality_report = f"""
================ DATA QUALITY REPORT ================

Total Rows: {len(data)}

Total Columns: {len(data.columns)}

Missing Values: {missing_values}

Duplicate Rows: {duplicate_rows}

Data Quality Status:
"""

    if missing_values == 0 and duplicate_rows == 0:

        quality_report += (
            "Clean dataset - no missing or duplicate records detected."
        )

    else:

        quality_report += (
            "Dataset requires additional data cleaning."
        )

    quality_report += "\n===================================================="

    return quality_report


# ============================================================
# STEP 60: AUTOMATED BUSINESS REPORT
# ============================================================

def generate_business_report(data):

    summary = generate_dataset_summary(data)

    quality = generate_data_quality_report(data)

    report = f"""
============================================================
                AI DATA ANALYST REPORT
============================================================

{summary}

{quality}

Business Insights:

- Revenue and profit performance were calculated from
  the uploaded dataset.

- The highest revenue-generating product was identified.

- The highest revenue-generating region was identified.

- Data quality was checked for missing and duplicate values.

============================================================
"""

    return report