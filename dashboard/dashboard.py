import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Bike Sharing Expert Dashboard",
    page_icon="🚲",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "day.csv")

day_df = pd.read_csv(DATA_PATH)
day_df["dteday"] = pd.to_datetime(day_df["dteday"])
day_df["year"] = day_df["dteday"].dt.year
day_df["month"] = day_df["dteday"].dt.month_name()

# Mapping kategori (sama dengan Colab)
day_df["season"] = day_df["season"].map({
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter"
})

day_df["weathersit"] = day_df["weathersit"].map({
    1: "Clear",
    2: "Mist",
    3: "Light Rain/Snow",
    4: "Heavy Rain/Snow"
})

day_df["workingday"] = day_df["workingday"].map({
    0: "Weekend/Holiday",
    1: "Working Day"
})

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Filter Data")

year_filter = st.sidebar.selectbox("Pilih Tahun", sorted(day_df["year"].unique()))
season_filter = st.sidebar.multiselect("Pilih Musim",
                                       day_df["season"].unique(),
                                       default=day_df["season"].unique())
workingday_filter = st.sidebar.multiselect("Pilih Jenis Hari",
                                           day_df["workingday"].unique(),
                                           default=day_df["workingday"].unique())
cluster_k = st.sidebar.slider("Jumlah Cluster (KMeans)", 2, 6, 3)

year_df = day_df[day_df["year"] == year_filter]
min_date = year_df["dteday"].min()
max_date = year_df["dteday"].max()

date_range = st.sidebar.date_input(
    "Rentang Tanggal",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

start_date, end_date = date_range

filtered_df = day_df[
    (day_df["year"] == year_filter) &
    (day_df["season"].isin(season_filter)) &
    (day_df["workingday"].isin(workingday_filter)) &
    (day_df["dteday"] >= pd.to_datetime(start_date)) &
    (day_df["dteday"] <= pd.to_datetime(end_date))
].copy()

# =========================
# KPI
# =========================
st.title("🚲 Bike Sharing 5-Star Analytics Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Peminjaman", f"{int(filtered_df['cnt'].sum()):,}")
col2.metric("Rata-rata Harian", f"{int(filtered_df['cnt'].mean()):,}")
col3.metric("Hari Observasi", len(filtered_df))

# =========================
# TREND + MA7
# =========================
st.subheader("Trend Peminjaman + Moving Average")

ts = filtered_df.groupby("dteday")["cnt"].sum().reset_index()
ts["MA7"] = ts["cnt"].rolling(7).mean()

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=ts["dteday"], y=ts["cnt"], name="Actual"))
fig_trend.add_trace(go.Scatter(x=ts["dteday"], y=ts["MA7"], name="MA7"))
st.plotly_chart(fig_trend, use_container_width=True)

# =========================
# MONTHLY BARCHART
# =========================
st.subheader("Rata-rata Rental per Bulan")

monthly_avg = filtered_df.groupby("month")["cnt"].mean().reset_index()
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

monthly_avg["month"] = pd.Categorical(monthly_avg["month"],
                                      categories=month_order,
                                      ordered=True)
monthly_avg = monthly_avg.sort_values("month")

fig_month = px.bar(monthly_avg, x="month", y="cnt", text="cnt")
fig_month.update_traces(texttemplate="%{text:.0f}", textposition="outside")
st.plotly_chart(fig_month, use_container_width=True)

# =========================
# SEASON INDEX
# =========================
st.subheader("Seasonal Index")

season_avg = filtered_df.groupby("season")["cnt"].mean()
overall = filtered_df["cnt"].mean()
season_index = (season_avg / overall).reset_index()
season_index.columns = ["season","index"]

fig_season = px.bar(season_index, x="season", y="index", text="index", color="season")
fig_season.update_traces(texttemplate="%{text:.2f}", textposition="outside")
st.plotly_chart(fig_season, use_container_width=True)

# =========================
# WEATHER VS RENTAL (BAR)
# =========================
st.subheader("Rata-rata Rental per Cuaca")

weather_avg = filtered_df.groupby("weathersit")["cnt"].mean().reset_index()
fig_weather = px.bar(weather_avg, x="weathersit", y="cnt", text="cnt", color="weathersit")
fig_weather.update_traces(texttemplate="%{text:.0f}", textposition="outside")
st.plotly_chart(fig_weather, use_container_width=True)

# =========================
# SCATTER MULTI VARIABLE
# =========================
st.subheader("Temperature vs Rental (Colored by Season)")

fig_scatter = px.scatter(filtered_df,
                         x="temp",
                         y="cnt",
                         color="season",
                         size="cnt",
                         opacity=0.7)
st.plotly_chart(fig_scatter, use_container_width=True)

# =========================
# REGRESSION
# =========================
st.subheader("Linear Regression Temp → Rental")

if len(filtered_df) > 10:
    model = LinearRegression()
    X = filtered_df[["temp"]]
    y = filtered_df["cnt"]
    model.fit(X,y)
    y_pred = model.predict(X)

    fig_reg = px.scatter(filtered_df, x="temp", y="cnt")
    fig_reg.add_trace(go.Scatter(x=filtered_df["temp"], y=y_pred, name="Regression"))
    st.plotly_chart(fig_reg, use_container_width=True)

    st.success(f"Slope: {model.coef_[0]:.2f}")

# =========================
# ANOMALY DETECTION
# =========================
st.subheader("Deteksi Anomali (Z-Score)")

filtered_df["z"] = (filtered_df["cnt"] - filtered_df["cnt"].mean()) / filtered_df["cnt"].std()
anomaly = filtered_df[np.abs(filtered_df["z"]) > 2]

fig_anom = px.scatter(filtered_df, x="dteday", y="cnt")
if not anomaly.empty:
    fig_anom.add_trace(go.Scatter(x=anomaly["dteday"],
                                  y=anomaly["cnt"],
                                  mode="markers",
                                  name="Anomaly"))
st.plotly_chart(fig_anom, use_container_width=True)

# =========================
# CLUSTERING
# =========================
st.subheader("Clustering (KMeans)")

features = filtered_df[["temp","hum","windspeed","cnt"]]
scaled = StandardScaler().fit_transform(features)

if len(filtered_df) > 5:
    kmeans = KMeans(n_clusters=cluster_k, random_state=42)
    filtered_df["Cluster"] = kmeans.fit_predict(scaled).astype(str)

    fig_cluster = px.scatter(filtered_df,
                             x="temp",
                             y="cnt",
                             color="Cluster",
                             size="cnt")
    st.plotly_chart(fig_cluster, use_container_width=True)

    cluster_profile = filtered_df.groupby("Cluster")[["cnt","temp","hum"]].mean().round(2)
    st.write("Cluster Profiling:")
    st.dataframe(cluster_profile)

# =========================
# CORRELATION RANKING
# =========================
st.subheader("Ranking Korelasi")

corr = filtered_df[["temp","hum","windspeed","cnt"]].corr()
corr_rank = corr["cnt"].drop("cnt").abs().sort_values(ascending=False).reset_index()
corr_rank.columns = ["Variabel","Korelasi"]

fig_corr = px.bar(corr_rank, x="Variabel", y="Korelasi", text="Korelasi")
fig_corr.update_traces(texttemplate="%{text:.2f}", textposition="outside")
st.plotly_chart(fig_corr, use_container_width=True)

# =========================
# AUTO INSIGHT
# =========================
st.subheader("Insight Otomatis")

highest_season = filtered_df.groupby("season")["cnt"].mean().idxmax()
lowest_season = filtered_df.groupby("season")["cnt"].mean().idxmin()

st.info(f"""
• Musim tertinggi: {highest_season}  
• Musim terendah: {lowest_season}  
• Jumlah anomali terdeteksi: {len(anomaly)}  
• Total data dianalisis: {len(filtered_df)} hari  
""")