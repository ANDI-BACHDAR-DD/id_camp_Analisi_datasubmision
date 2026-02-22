import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# Mapping kategori
day_df["season"] = day_df["season"].map({
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter"
})

day_df["workingday"] = day_df["workingday"].map({
    0: "Weekend/Holiday",
    1: "Working Day"
})

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Filter Data")

year_filter = st.sidebar.selectbox(
    "Pilih Tahun",
    sorted(day_df["year"].unique())
)

season_filter = st.sidebar.multiselect(
    "Pilih Musim",
    day_df["season"].unique(),
    default=day_df["season"].unique()
)

workingday_filter = st.sidebar.multiselect(
    "Pilih Jenis Hari",
    day_df["workingday"].unique(),
    default=day_df["workingday"].unique()
)

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
# TITLE & KPI
# =========================
st.title("🚲 Bike Sharing Expert Analytics Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Total Peminjaman", f"{int(filtered_df['cnt'].sum()):,}")
col2.metric("Rata-rata Harian", f"{int(filtered_df['cnt'].mean()):,}")
col3.metric("Hari Observasi", len(filtered_df))

# =========================
# TIME SERIES + MOVING AVG
# =========================
st.subheader("Trend Peminjaman Harian + Moving Average")

time_series = filtered_df.groupby("dteday")["cnt"].sum().reset_index()
time_series["MA7"] = time_series["cnt"].rolling(7).mean()

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=time_series["dteday"], y=time_series["cnt"], name="Actual"))
fig_trend.add_trace(go.Scatter(x=time_series["dteday"], y=time_series["MA7"], name="MA 7 Hari"))

st.plotly_chart(fig_trend, use_container_width=True)

# =========================
# SCATTER PLOTS
# =========================
colA, colB = st.columns(2)

with colA:
    st.subheader("Temp vs Rentals")
    fig_temp = px.scatter(filtered_df, x="temp", y="cnt",
                          color="season", size="cnt", opacity=0.7)
    st.plotly_chart(fig_temp, use_container_width=True)

with colB:
    st.subheader("Humidity vs Rentals")
    fig_hum = px.scatter(filtered_df, x="hum", y="cnt",
                         color="workingday", size="cnt", opacity=0.7)
    st.plotly_chart(fig_hum, use_container_width=True)

# =========================
# DISTRIBUTION
# =========================
colC, colD = st.columns(2)

with colC:
    st.subheader("Box Plot per Musim")
    fig_box = px.box(filtered_df, x="season", y="cnt", color="season")
    st.plotly_chart(fig_box, use_container_width=True)

with colD:
    st.subheader("Violin Plot per Musim")
    fig_violin = px.violin(filtered_df, x="season", y="cnt",
                           color="season", box=True)
    st.plotly_chart(fig_violin, use_container_width=True)

st.subheader("Histogram Distribusi Rentals")
fig_hist = px.histogram(filtered_df, x="cnt", nbins=30, marginal="box")
st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# HEATMAP
# =========================
st.subheader("Heatmap Korelasi")

corr = filtered_df[["temp", "hum", "windspeed", "cnt"]].corr()

fig_heat = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale="Blues",
    text=corr.values.round(2),
    texttemplate="%{text}"
))

st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# TOP 10
# =========================
st.subheader("Top 10 Hari Tertinggi")

top10 = filtered_df.sort_values("cnt", ascending=False).head(10)
fig_top = px.bar(top10, x="dteday", y="cnt", text="cnt")
fig_top.update_traces(textposition="outside")
st.plotly_chart(fig_top, use_container_width=True)

# =========================
# ELBOW METHOD
# =========================
st.subheader("Elbow Method")

features = filtered_df[["temp", "hum", "windspeed", "cnt"]]
scaled = StandardScaler().fit_transform(features)

inertia = []
K_range = range(1, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled)
    inertia.append(km.inertia_)

fig_elbow = px.line(x=list(K_range), y=inertia, markers=True)
st.plotly_chart(fig_elbow, use_container_width=True)

# =========================
# CLUSTERING
# =========================
st.subheader("Clustering Analysis")

if len(filtered_df) > 5:
    kmeans = KMeans(n_clusters=cluster_k, random_state=42)
    filtered_df["Cluster"] = kmeans.fit_predict(scaled).astype(str)

    fig_cluster = px.scatter(filtered_df,
                             x="temp",
                             y="cnt",
                             color="Cluster",
                             size="cnt",
                             hover_data=["season", "hum", "windspeed"])
    st.plotly_chart(fig_cluster, use_container_width=True)

    cluster_summary = filtered_df.groupby("Cluster")["cnt"].mean().reset_index()

    st.write("Rata-rata Peminjaman per Cluster:")
    st.dataframe(cluster_summary)

# =========================
# AUTO INSIGHT
# =========================
st.subheader("Insight Otomatis")

highest_season = filtered_df.groupby("season")["cnt"].mean().idxmax()
lowest_season = filtered_df.groupby("season")["cnt"].mean().idxmin()

st.info(f"""
• Musim dengan rata-rata peminjaman tertinggi: **{highest_season}**
• Musim dengan rata-rata terendah: **{lowest_season}**
• Total data dianalisis: {len(filtered_df)} hari
""")