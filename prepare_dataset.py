"""
Aquatic E-Commerce: cleaning + feature engineering pipeline
Usage: python prepare_dataset.py Aquatic_E-Comerce.csv outputs_dir
Produces:
  1) aquatic_demand_ml_dataset.csv          (monthly demand per product_line x location, for ML)
  2) aquatic_transactions_clean_10k.csv     (10,000 stratified cleaned transactions, for EDA/dashboard)
  3) cleaning_log.json                      (row counts at each step)
"""
import sys, json
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "Aquatic_E-Comerce.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
SEED = 42
rng = np.random.default_rng(SEED)
log = {}

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv(SRC)
log["raw_rows"] = len(df)

# ---------------------------------------------------------------- 2. CLEAN
# 2a. text standardisation
for c in ["category", "subcategory", "brand", "location", "device", "payment_method", "delivery_status"]:
    df[c] = df[c].astype(str).str.strip()
df["category"] = df["category"].replace({"FishFood": "Fish Food", "Aquatic plants": "Aquatic Plants"})
df["subcategory"] = df["subcategory"].replace({"Installation accessories": "Installation Accessories"})

# 2b. dates (source format is dd-mm-yyyy)
df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="%d-%m-%Y", errors="coerce")
log["bad_dates_dropped"] = int(df["purchase_date"].isna().sum())
df = df.dropna(subset=["purchase_date"])

# 2c. duplicates & nulls
log["duplicate_rows_dropped"] = int(df.duplicated().sum())
df = df.drop_duplicates()
log["rows_with_nulls_dropped"] = int(df.isnull().any(axis=1).sum())
df = df.dropna()

# 2d. business-rule validation
bad = (
    (df.price <= 0) | (df.discount < 0) | (df.discount > 100)
    | (df.rating < 1) | (df.rating > 5) | (df.stock < 0) | (df.review_count < 0)
)
log["invalid_rule_rows_dropped"] = int(bad.sum())
df = df[~bad]

# final_price must equal price * (1 - discount/100); source differs only by rounding
df["final_price"] = (df["price"] * (1 - df["discount"] / 100)).round(2)

# 2e. off-domain subcategories (not aquatic products)
OFF_DOMAIN = ["Headphones", "Kitchen"]
mask_off = df["subcategory"].isin(OFF_DOMAIN)
log["off_domain_rows_dropped"] = int(mask_off.sum())
df = df[~mask_off].copy()

# 2f. rating with zero reviews is not a real rating
df["has_reviews"] = (df["review_count"] > 0).astype(int)
log["rating_set_missing_zero_reviews"] = int((df["has_reviews"] == 0).sum())
df.loc[df["has_reviews"] == 0, "rating"] = np.nan
df["rating"] = df["rating"].fillna(df.groupby("subcategory")["rating"].transform("median"))

# 2g. heavy-tailed review_count: winsorise at 99.9th percentile (price is NOT trimmed:
#     it is bounded per category; a global IQR rule would wrongly drop valid equipment prices)
cap = df["review_count"].quantile(0.999)
log["review_count_cap"] = float(cap)
df["review_count"] = df["review_count"].clip(upper=cap)

log["clean_rows"] = len(df)

# ---------------------------------------------------------------- 3. TRANSACTION-LEVEL FEATURES
df["product_line"] = df["subcategory"] + " | " + df["brand"]
df["year"] = df["purchase_date"].dt.year
df["month_num"] = df["purchase_date"].dt.month
df["quarter"] = df["purchase_date"].dt.quarter
df["week_of_year"] = df["purchase_date"].dt.isocalendar().week.astype(int)
df["day_of_week"] = df["purchase_date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["days_in_month"] = df["purchase_date"].dt.days_in_month
df["year_month"] = df["purchase_date"].dt.to_period("M")
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
          6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-Monsoon", 11: "Post-Monsoon"}
df["season"] = df["month_num"].map(SEASON)

df["discount_amount"] = (df["price"] - df["final_price"]).round(2)
df["log_price"] = np.log1p(df["price"]).round(4)
df["is_high_discount"] = (df["discount"] >= 40).astype(int)
df["discount_band"] = pd.cut(df["discount"], [0, 15, 30, 45, 100], labels=["Low", "Medium", "High", "Very High"]).astype(str)
df["price_band"] = df.groupby("category")["price"].transform(
    lambda s: pd.qcut(s, 3, labels=["Budget", "Mid", "Premium"]).astype(str))
df["is_returned"] = df["is_returned"].astype(int)
df["is_delayed"] = (df["delivery_status"] == "Delayed").astype(int)
df["is_out_of_stock"] = (df["stock"] == 0).astype(int)
df["is_low_stock"] = (df["stock"] < 50).astype(int)
df["fast_shipping"] = (df["shipping_time_days"] <= 2).astype(int)
# 'In Transit' on orders much older than the data end date is logically inconsistent
END = df["purchase_date"].max()
df["status_inconsistent"] = ((df["delivery_status"] == "In Transit") &
                             ((END - df["purchase_date"]).dt.days > 14)).astype(int)
log["in_transit_older_than_14d_flagged"] = int(df["status_inconsistent"].sum())
# customer feature (computed on the FULL cleaned data before sampling)
orders_per_user = df.groupby("user_id")["user_id"].transform("size")
df["customer_order_count"] = orders_per_user
df["is_repeat_customer"] = (orders_per_user > 1).astype(int)

# ---------------------------------------------------------------- 4. MONTHLY ML TABLE
# keep only full months (data starts on 31-Mar-2024 -> March 2024 has a single day)
full = df[df["year_month"] >= "2024-04"].copy()
full["month_start"] = full["purchase_date"].dt.to_period("M").dt.to_timestamp()
keys = ["category", "subcategory", "brand", "product_line", "location", "month_start"]

agg = full.groupby(keys).agg(
    demand=("product_id", "size"),                       # TARGET: number of transactions
    avg_price=("price", "mean"),
    avg_discount=("discount", "mean"),
    avg_final_price=("final_price", "mean"),
    avg_stock=("stock", "mean"),
    low_stock_share=("is_low_stock", "mean"),
    avg_rating=("rating", "mean"),
    avg_review_count=("review_count", "mean"),
    return_rate=("is_returned", "mean"),
    delay_rate=("is_delayed", "mean"),
    avg_shipping_days=("shipping_time_days", "mean"),
    ana_revenue=("final_price", "sum"),
    ana_unique_customers=("user_id", "nunique"),
).reset_index()

# complete the grid so months with zero orders appear as demand = 0
grid = pd.MultiIndex.from_product(
    [full[["category", "subcategory", "brand", "product_line"]].drop_duplicates().itertuples(index=False, name=None),
     full["location"].unique(), sorted(full["month_start"].unique())]).to_frame(index=False)
grid = pd.DataFrame(grid[0].tolist(), columns=["category", "subcategory", "brand", "product_line"]).assign(
    location=grid[1].values, month_start=grid[2].values)
agg = grid.merge(agg, on=keys, how="left")
log["empty_cells_filled_with_zero"] = int(agg["demand"].isna().sum())
agg["demand"] = agg["demand"].fillna(0).astype(int)
agg = agg.sort_values(["product_line", "location", "month_start"]).reset_index(drop=True)
grp = agg.groupby(["product_line", "location"], sort=False)

# carry-forward then group-median for any month without transactions
num_cols = ["avg_price", "avg_discount", "avg_final_price", "avg_stock", "low_stock_share", "avg_rating",
            "avg_review_count", "return_rate", "delay_rate", "avg_shipping_days"]
for c in num_cols:
    agg[c] = grp[c].ffill()
    agg[c] = agg[c].fillna(agg.groupby("product_line")[c].transform("median"))
for c in ["ana_revenue", "ana_unique_customers"]:
    agg[c] = agg[c].fillna(0)
grp = agg.groupby(["product_line", "location"], sort=False)

# calendar features
m = agg["month_start"]
agg["year"] = m.dt.year
agg["month_num"] = m.dt.month
agg["quarter"] = m.dt.quarter
agg["days_in_month"] = m.dt.days_in_month
agg["season"] = agg["month_num"].map(SEASON)
agg["month_index"] = (m.dt.year - 2024) * 12 + m.dt.month - 3      # Apr-2024 = 1
agg["ana_demand_per_day"] = (agg["demand"] / agg["days_in_month"]).round(3)

# lag / rolling features (all use only PAST months -> no leakage)
d = grp["demand"]
agg["demand_lag_1"] = d.shift(1)
agg["demand_lag_2"] = d.shift(2)
agg["demand_lag_3"] = d.shift(3)
agg["demand_roll_mean_3"] = d.transform(lambda x: x.shift(1).rolling(3).mean())
agg["demand_roll_std_3"] = d.transform(lambda x: x.shift(1).rolling(3).std())
agg["demand_roll_mean_6"] = d.transform(lambda x: x.shift(1).rolling(6).mean())
agg["demand_expanding_mean"] = d.transform(lambda x: x.shift(1).expanding().mean())
agg["demand_growth_1"] = ((agg["demand_lag_1"] - agg["demand_lag_2"]) / agg["demand_lag_2"].replace(0, np.nan)).round(4)
# demand per day of previous month (removes the 28/30/31-day effect)
agg["demand_per_day_lag_1"] = (agg["demand_lag_1"] / grp["days_in_month"].shift(1)).round(3)

# previous-month versions of outcome-type columns (safe to use as features)
for c in ["avg_stock", "low_stock_share", "avg_rating", "avg_review_count", "return_rate", "delay_rate"]:
    agg[f"{c}_lag_1"] = grp[c].shift(1)
# stock-and-price planning inputs are kept current-month (treated as known plan) -> see report

# slow-moving flag: bottom 10% of 3-month rolling demand among all lines in the same month
agg["is_slow_moving"] = (agg["demand_roll_mean_3"] <
                         agg.groupby("month_start")["demand_roll_mean_3"].transform(lambda x: x.quantile(0.10))).astype(int)

# drop warm-up months that have no complete lag window (first 6 months)
ml = agg.dropna(subset=["demand_roll_mean_6", "demand_lag_1", "demand_roll_std_3", "demand_growth_1"]).copy()
log["ml_rows_dropped_warmup"] = int(len(agg) - len(ml))

# SIMULATED inventory (the source 'stock' is a random per-transaction snapshot, not a ledger)
cover = rng.uniform(0.8, 1.8, len(ml))
ml["sim_current_stock"] = np.round(ml["demand_lag_1"] * cover).astype(int)
ratio = ml["demand"] / ml["sim_current_stock"].replace(0, np.nan)
ml["sim_stockout_risk"] = np.select([ratio.fillna(9) >= 1.0, ratio >= 0.7], ["High", "Medium"], default="Low")
ml["sim_reorder_qty"] = (ml["demand"] - ml["sim_current_stock"]).clip(lower=0).astype(int)

# label codes (for tree models; use one-hot in a sklearn Pipeline for Linear Regression)
for c in ["category", "subcategory", "brand", "location", "season"]:
    ml[f"{c}_code"] = ml[c].astype("category").cat.codes
ml["product_line_id"] = ml["product_line"].astype("category").cat.codes + 1

ml["split"] = np.where(ml["month_index"] <= ml["month_index"].max() - 3, "train", "test")   # last 3 months = test
for c in ml.select_dtypes("float").columns:
    ml[c] = ml[c].round(4)
ml["month_start"] = ml["month_start"].dt.strftime("%Y-%m-%d")

col_order = (
    ["month_start", "category", "subcategory", "brand", "product_line", "product_line_id", "location", "split", "demand"]
    + ["year", "month_num", "quarter", "days_in_month", "month_index", "season"]
    + ["category_code", "subcategory_code", "brand_code", "location_code", "season_code"]
    + ["demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_roll_mean_3", "demand_roll_std_3",
       "demand_roll_mean_6", "demand_expanding_mean", "demand_growth_1", "demand_per_day_lag_1"]
    + ["avg_price", "avg_discount", "avg_final_price", "avg_stock", "low_stock_share", "avg_rating",
       "avg_review_count", "return_rate", "delay_rate", "avg_shipping_days"]
    + [f"{c}_lag_1" for c in ["avg_stock", "low_stock_share", "avg_rating", "avg_review_count", "return_rate", "delay_rate"]]
    + ["is_slow_moving", "sim_current_stock", "sim_stockout_risk", "sim_reorder_qty"]
    + ["ana_revenue", "ana_unique_customers", "ana_demand_per_day"]
)
ml = ml[col_order]
ml.to_csv(f"{OUT}/aquatic_demand_ml_dataset.csv", index=False)
log["ml_rows"] = len(ml)

# ---------------------------------------------------------------- 5. 10,000-ROW TRANSACTION SAMPLE
N = 10_000
strata = df.groupby(["subcategory", "year_month"]).size()
quota = strata / strata.sum() * N
alloc = np.floor(quota).astype(int)
for k in (quota - alloc).sort_values(ascending=False).index[: N - alloc.sum()]:
    alloc[k] += 1
parts = [g.sample(n=int(alloc[k]), random_state=SEED) for k, g in df.groupby(["subcategory", "year_month"]) if alloc[k] > 0]
tx = pd.concat(parts).sort_values("purchase_date").reset_index(drop=True)
tx["purchase_date"] = tx["purchase_date"].dt.strftime("%Y-%m-%d")
tx = tx.drop(columns=["year_month", "seller_id"])
tx["transaction_id"] = ["T" + str(i + 1).zfill(5) for i in range(len(tx))]
first = ["transaction_id", "purchase_date", "user_id", "product_id", "category", "subcategory", "brand", "product_line"]
tx = tx[first + [c for c in tx.columns if c not in first]]
for c in tx.select_dtypes("float").columns:
    tx[c] = tx[c].round(4)
tx.to_csv(f"{OUT}/aquatic_transactions_clean_10k.csv", index=False)
log["sample_rows"] = len(tx)

with open(f"{OUT}/cleaning_log.json", "w") as f:
    json.dump(log, f, indent=2)
print(json.dumps(log, indent=2))
