import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# AquaStock Intelligence
# ML-Powered Demand Forecasting & Inventory Decision System
# ============================================================

st.set_page_config(
    page_title="AquaStock Intelligence",
    page_icon="🐟",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BUSINESS_FILE = DATA / "inventory_business_output.csv"
DATA_FILE = DATA / "aquatic_demand_ml_dataset.csv"
MODEL_FILE = BASE / "model.pkl"

st.markdown("""
<style>
.main-title{font-size:2.2rem;font-weight:750}
.subtitle{color:#6b7280;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)


def find_col(df, names):
    exact = {str(c).lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in exact:
            return exact[n.lower()]
    for c in df.columns:
        for n in names:
            if n.lower() in str(c).lower():
                return c
    return None


@st.cache_data
def read_csv(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return None


@st.cache_resource
def read_model(path):
    if Path(path).exists():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def number(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "0"


def add_business_logic(df):
    df = df.copy()

    pred = find_col(df, [
        "predicted_demand", "prediction", "forecast_demand",
        "predicted demand"
    ])
    demand = find_col(df, ["demand"])
    stock = find_col(df, [
        "current_stock", "stock", "inventory",
        "stock_level", "last_month_stock"
    ])

    if pred and pred != "predicted_demand":
        df["predicted_demand"] = pd.to_numeric(df[pred], errors="coerce")
    elif "predicted_demand" not in df.columns:
        df["predicted_demand"] = pd.to_numeric(
            df[demand], errors="coerce"
        ) if demand else 0

    if stock and stock != "current_stock":
        df["current_stock"] = pd.to_numeric(df[stock], errors="coerce")
    elif "current_stock" not in df.columns:
        df["current_stock"] = 0

    for c in ["predicted_demand", "current_stock"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["stock_gap"] = df["current_stock"] - df["predicted_demand"]
    df["stockout_risk"] = df["current_stock"] < df["predicted_demand"]
    df["safety_stock"] = df["predicted_demand"] * 0.10
    df["target_stock"] = df["predicted_demand"] + df["safety_stock"]
    df["reorder_quantity"] = (
        df["target_stock"] - df["current_stock"]
    ).clip(lower=0).apply(lambda x: math.ceil(x))

    df["stock_coverage_months"] = np.where(
        df["predicted_demand"] > 0,
        df["current_stock"] / df["predicted_demand"],
        np.nan
    )

    if "inventory_status" not in df.columns:
        df["inventory_status"] = np.select(
            [
                df["stock_coverage_months"] < 1,
                df["stock_coverage_months"] <= 2
            ],
            ["Stockout Risk", "Healthy"],
            default="Overstocked"
        )

    product = find_col(df, [
        "product", "product_name", "product_line",
        "product_type", "category"
    ])
    df["product_display"] = (
        df[product].astype(str)
        if product else
        [f"Product {i+1}" for i in range(len(df))]
    )
    return df


# Prefer Phase 4 output. Otherwise build the same logic from the main CSV.
df = read_csv(BUSINESS_FILE)
source = "Phase 4 business output"

if df is None:
    df = read_csv(DATA_FILE)
    source = "Main dataset + business calculations"

if df is None:
    st.error(
    "CSV not found. Please check that the required CSV files "
    "are present in the project root directory."
    )
    st.stop()

df = add_business_logic(df)
model = read_model(MODEL_FILE)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("🐟 AquaStock")
st.sidebar.caption("Inventory Intelligence System")

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Dashboard",
        "Demand Forecast",
        "Inventory Health",
        "Smart Reorder",
        "Slow-Moving Products",
        "Product Intelligence",
        "What-If Simulator",
        "Model Performance"
    ]
)

st.sidebar.divider()
st.sidebar.caption(f"Data: {source}")
st.sidebar.caption("Safety stock assumption: 10%")

# Filters
st.sidebar.subheader("🔎 Filters")

for label, colnames in [
    ("Product", ["product", "product_name", "product_line",
                 "product_type", "category"]),
    ("City", ["city", "location"]),
    ("Brand", ["brand", "brand_name"])
]:
    col = find_col(df, colnames)
    if col:
        values = sorted(df[col].dropna().astype(str).unique())
        selected = st.sidebar.multiselect(label, values)
        if selected:
            df = df[df[col].astype(str).isin(selected)]

statuses = sorted(df["inventory_status"].dropna().astype(str).unique())
selected_status = st.sidebar.multiselect("Inventory Status", statuses)
if selected_status:
    df = df[df["inventory_status"].isin(selected_status)]

if df.empty:
    st.warning("No records match the selected filters.")
    st.stop()


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================
if page == "Executive Dashboard":
    st.markdown(
        '<div class="main-title">🐟 AquaStock Intelligence</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">ML-Powered Demand Forecasting & Inventory Decision System</div>',
        unsafe_allow_html=True
    )

    risk = int((df["inventory_status"] == "Stockout Risk").sum())
    healthy = int((df["inventory_status"] == "Healthy").sum())
    over = int((df["inventory_status"] == "Overstocked").sum())

    a,b,c,d,e = st.columns(5)
    a.metric("Products", number(len(df)))
    b.metric("Predicted Demand", number(df.predicted_demand.sum()))
    c.metric("🔴 Stockout Risk", number(risk))
    d.metric("🟡 Overstocked", number(over))
    e.metric("Recommended Reorder", number(df.reorder_quantity.sum()))

    st.divider()

    if risk:
        st.warning(
            f"{risk} product records may not have enough stock "
            "to meet predicted demand."
        )
    else:
        st.success("No stockout-risk records found under the current filters.")

    c1,c2 = st.columns(2)

    with c1:
        st.subheader("Inventory Status")
        counts = (
            df.inventory_status.value_counts()
            .reindex(["Stockout Risk","Healthy","Overstocked"])
            .fillna(0)
        )
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)

    with c2:
        st.subheader("Predicted Demand vs Current Stock")
        vals = [
            df.predicted_demand.sum(),
            df.current_stock.sum()
        ]
        fig, ax = plt.subplots()
        ax.bar(["Predicted Demand","Current Stock"], vals)
        ax.set_ylabel("Units")
        st.pyplot(fig)

    st.subheader("🚨 Highest Reorder Priorities")
    priority = df.sort_values(
        ["reorder_quantity","stock_gap"],
        ascending=[False,True]
    ).head(10)

    st.dataframe(
        priority[
            ["product_display","current_stock","predicted_demand",
             "stock_gap","reorder_quantity","inventory_status"]
        ].rename(columns={
            "product_display":"Product",
            "current_stock":"Current Stock",
            "predicted_demand":"Predicted Demand",
            "stock_gap":"Stock Gap",
            "reorder_quantity":"Reorder Qty",
            "inventory_status":"Status"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DEMAND FORECAST
# ============================================================
elif page == "Demand Forecast":
    st.title("📈 Demand Forecast")
    st.write(
        "The ML stage estimates future demand. This estimate becomes "
        "the starting point for inventory planning."
    )

    a,b,c = st.columns(3)
    a.metric("Total Predicted Demand", number(df.predicted_demand.sum()))
    b.metric("Average", f"{df.predicted_demand.mean():,.1f}")
    c.metric("Records", number(len(df)))

    date_col = find_col(df, [
        "date","month","order_date","transaction_date","year_month"
    ])

    if date_col:
        temp = df.copy()
        temp["_date"] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=["_date"])
        if not temp.empty:
            trend = temp.groupby("_date").predicted_demand.sum().sort_index()
            fig, ax = plt.subplots(figsize=(11,4))
            ax.plot(trend.index, trend.values, marker="o")
            ax.set_title("Predicted Demand Over Time")
            ax.set_ylabel("Demand")
            ax.grid(alpha=.25)
            st.pyplot(fig)
    else:
        summary = (
            df.groupby("product_display").predicted_demand.sum()
            .sort_values(ascending=False).head(15)
        )
        fig, ax = plt.subplots(figsize=(11,5))
        ax.bar(summary.index.astype(str), summary.values)
        ax.set_title("Top Products by Predicted Demand")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    st.subheader("Prediction Table")
    st.dataframe(
        df[
            ["product_display","predicted_demand","current_stock",
             "stock_gap","inventory_status"]
        ].rename(columns={
            "product_display":"Product",
            "predicted_demand":"Predicted Demand",
            "current_stock":"Current Stock",
            "stock_gap":"Stock Gap",
            "inventory_status":"Status"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INVENTORY HEALTH
# ============================================================
elif page == "Inventory Health":
    st.title("📦 Inventory Health")
    st.write(
        "We compare current inventory with predicted demand and classify "
        "each record as Stockout Risk, Healthy, or Overstocked."
    )

    counts = (
        df.inventory_status.value_counts()
        .reindex(["Stockout Risk","Healthy","Overstocked"])
        .fillna(0)
    )

    a,b,c = st.columns(3)
    a.metric("🔴 Stockout Risk", int(counts["Stockout Risk"]))
    b.metric("🟢 Healthy", int(counts["Healthy"]))
    c.metric("🟡 Overstocked", int(counts["Overstocked"]))

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Records")
    ax.set_title("Inventory Status Distribution")
    st.pyplot(fig)

    st.subheader("Stock Coverage")
    view = df[
        ["product_display","current_stock","predicted_demand",
         "stock_coverage_months","inventory_status"]
    ].sort_values("stock_coverage_months", na_position="first")

    st.dataframe(
        view.rename(columns={
            "product_display":"Product",
            "current_stock":"Current Stock",
            "predicted_demand":"Predicted Demand",
            "stock_coverage_months":"Coverage (Months)",
            "inventory_status":"Status"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SMART REORDER
# ============================================================
elif page == "Smart Reorder":
    st.title("🛒 Smart Reorder Recommendations")
    st.write(
        "Recommended reorder = Target Stock − Current Stock, where "
        "Target Stock = Predicted Demand + 10% Safety Stock."
    )

    reorder = df[df.reorder_quantity > 0].sort_values(
        ["reorder_quantity","stock_gap"],
        ascending=[False,True]
    )

    a,b = st.columns(2)
    a.metric("Records to Reorder", number(len(reorder)))
    b.metric("Total Units", number(reorder.reorder_quantity.sum()))

    if reorder.empty:
        st.success("No reorder is recommended for the current filters.")
    else:
        cols = [
            "product_display","current_stock","predicted_demand",
            "safety_stock","target_stock","reorder_quantity",
            "inventory_status"
        ]
        st.dataframe(
            reorder[cols].rename(columns={
                "product_display":"Product",
                "current_stock":"Current Stock",
                "predicted_demand":"Predicted Demand",
                "safety_stock":"Safety Stock",
                "target_stock":"Target Stock",
                "reorder_quantity":"Reorder Quantity",
                "inventory_status":"Status"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Reorder List",
            reorder[cols].to_csv(index=False).encode(),
            "aquastock_reorder_recommendations.csv",
            "text/csv"
        )


# ============================================================
# SLOW-MOVING PRODUCTS
# ============================================================
elif page == "Slow-Moving Products":
    st.title("🐢 Slow-Moving Products")
    st.write(
        "We identify relatively low-demand products so the business can "
        "avoid unnecessary additional inventory."
    )

    slow_flag = find_col(df, [
        "slow_moving","slow-moving","slow_moving_product"
    ])

    if slow_flag:
        slow = df[df[slow_flag].astype(str).str.lower().isin(
            ["true","yes","1","slow moving","slow-moving"]
        )]
    else:
        threshold = df.predicted_demand.quantile(.25)
        slow = df[df.predicted_demand <= threshold]

    a,b = st.columns(2)
    a.metric("Slow-Moving Records", len(slow))
    b.metric(
        "Average Demand",
        f"{slow.predicted_demand.mean():,.1f}" if len(slow) else "0"
    )

    st.dataframe(
        slow[
            ["product_display","predicted_demand","current_stock",
             "stock_coverage_months","inventory_status"]
        ].sort_values("predicted_demand").rename(columns={
            "product_display":"Product",
            "predicted_demand":"Predicted Demand",
            "current_stock":"Current Stock",
            "stock_coverage_months":"Coverage (Months)",
            "inventory_status":"Status"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Project assumption: if Phase 4 does not contain a slow-moving flag, "
        "the bottom 25% by predicted demand is treated as slow-moving."
    )


# ============================================================
# PRODUCT INTELLIGENCE
# ============================================================
elif page == "Product Intelligence":
    st.title("🔍 Product Intelligence")

    products = sorted(df.product_display.unique())
    selected = st.selectbox("Select Product", products)
    p = df[df.product_display == selected]

    demand = p.predicted_demand.sum()
    stock = p.current_stock.sum()
    safety = p.safety_stock.sum()
    target = p.target_stock.sum()
    reorder = p.reorder_quantity.sum()
    coverage = stock / demand if demand else np.nan

    status = (
        "🔴 STOCKOUT RISK" if coverage < 1
        else "🟢 HEALTHY" if coverage <= 2
        else "🟡 OVERSTOCKED"
    )

    a,b,c = st.columns(3)
    a.metric("Predicted Demand", number(demand))
    b.metric("Current Stock", number(stock))
    c.metric("Recommended Reorder", number(reorder))

    a,b,c = st.columns(3)
    a.metric("Safety Stock", number(safety))
    b.metric("Target Stock", number(target))
    c.metric(
        "Coverage",
        f"{coverage:.2f} months" if not np.isnan(coverage) else "N/A"
    )

    st.subheader(f"Status: {status}")

    gap = stock - demand
    if gap < 0:
        st.warning(
            f"Current stock is about {abs(gap):,.0f} units below predicted demand."
        )
    else:
        st.success(
            f"Current stock is about {gap:,.0f} units above predicted demand."
        )

    st.subheader("Why this recommendation?")
    st.info(
        f"Expected demand is {demand:,.0f} units. Current stock is "
        f"{stock:,.0f}. We keep {safety:,.0f} units as safety stock, "
        f"giving a target of {target:,.0f}. Therefore the recommended "
        f"reorder is {reorder:,.0f} units."
    )


# ============================================================
# WHAT-IF SIMULATOR
# ============================================================
elif page == "What-If Simulator":
    st.title("🔮 What-If Inventory Simulator")
    st.write(
        "Test how inventory requirements change if demand increases or decreases."
    )

    products = sorted(df.product_display.unique())
    selected = st.selectbox("Select Product", products, key="whatif_product")
    p = df[df.product_display == selected]

    base_demand = float(p.predicted_demand.sum())
    current_stock = float(p.current_stock.sum())

    change = st.slider(
        "Change in expected demand",
        -50, 50, 10, 5,
        format="%d%%"
    )

    scenario_demand = base_demand * (1 + change / 100)
    scenario_safety = scenario_demand * .10
    scenario_target = scenario_demand + scenario_safety
    scenario_reorder = max(scenario_target - current_stock, 0)

    a,b,c,d = st.columns(4)
    a.metric("Current Stock", number(current_stock))
    b.metric("Original Demand", number(base_demand))
    c.metric("Scenario Demand", number(scenario_demand))
    d.metric("Scenario Reorder", number(scenario_reorder))

    if scenario_reorder:
        st.warning(
            f"Under this scenario, approximately {scenario_reorder:,.0f} "
            "units should be ordered."
        )
    else:
        st.success("Current stock is sufficient for this scenario.")

    comparison = pd.DataFrame({
        "Metric":["Demand","Safety Stock","Target Stock",
                  "Current Stock","Recommended Reorder"],
        "Current Model":[
            base_demand, base_demand*.10, base_demand*1.10,
            current_stock, max(base_demand*1.10-current_stock,0)
        ],
        "Scenario":[
            scenario_demand, scenario_safety, scenario_target,
            current_stock, scenario_reorder
        ]
    })

    st.subheader("Scenario Comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.caption(
        "This is a business scenario simulation. It does not retrain the ML model."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.title("🤖 Model Performance")
    st.write(
        "Phase 3 compared simple baselines with Linear Regression and "
        "Random Forest using a time-based test period."
    )

    results = pd.DataFrame({
        "Model":[
            "Naive – Last Month",
            "Naive – 3 Month Average",
            "Linear Regression",
            "Random Forest"
        ],
        "MAE":[8.57,6.82,5.58,5.54],
        "RMSE":[10.96,8.74,7.09,7.05],
        "R²":[.707,.814,.877,.879]
    })

    st.dataframe(
        results.style.format({
            "MAE":"{:.2f}",
            "RMSE":"{:.2f}",
            "R²":"{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Best evaluated model: Random Forest — "
        "MAE 5.54 | RMSE 7.05 | R² 0.879"
    )

    a,b = st.columns(2)

    with a:
        fig, ax = plt.subplots()
        ax.bar(results.Model, results.RMSE)
        ax.set_title("RMSE Comparison")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    with b:
        fig, ax = plt.subplots()
        ax.bar(results.Model, results["R²"])
        ax.set_title("R² Comparison")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    st.subheader("How the model becomes a business solution")
    st.markdown("""
    **Historical Data**
    → **ML Demand Prediction**
    → **Stock Comparison**
    → **Risk Detection**
    → **Reorder Recommendation**
    → **Business Decision**
    """)


st.divider()
st.caption("AquaStock Intelligence • ML + Inventory Decision-Support Project")
