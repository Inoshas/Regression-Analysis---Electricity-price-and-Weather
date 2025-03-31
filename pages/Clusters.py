import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load Data
df = pd.read_csv("data/fmi_weather_and_price.csv", parse_dates=['Time'])
df["Month"] = df["Time"].dt.month  # Extract month
df_selected = df[["Temp", "Wind", "Price"]].dropna()  # Keep relevant features
df_selected["Wind"] = df_selected["Wind"]**3  # Apply wind power transformation

# Scale Data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_selected[["Temp", "Wind"]])

# K-Means Clustering
k = 4  # Define clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_selected["Cluster"] = kmeans.fit_predict(scaled_features)

# Regression Model for Each Cluster
regression_results = []
for cluster in range(k):
    cluster_data = df_selected[df_selected["Cluster"] == cluster]
    X = cluster_data[["Temp", "Wind"]]
    y = cluster_data["Price"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    regression_results.append({
        "Cluster": cluster,
        "Intercept": model.intercept_,
        "Temp Coefficient": model.coef_[0],
        "Wind Coefficient": model.coef_[1]
    })

# Convert results to DataFrame
regression_df = pd.DataFrame(regression_results)

# **Streamlit App**
st.title("Electricity Price Analysis: Clustering & Regression")

# **Cluster Visualization**
st.subheader("K-Means Clustering (Temp & Wind)")
fig_cluster = px.scatter(df_selected, x="Temp", y="Wind", color=df_selected["Cluster"].astype(str),
                         title="Temperature vs Wind Speed Clustering",
                         labels={"Temp": "Temperature (°C)", "Wind": "Wind Speed³"},
                         hover_data=["Price"])
st.plotly_chart(fig_cluster)

# **Regression Results Table**
st.subheader("Linear Regression Coefficients per Cluster")
st.dataframe(regression_df)

# **Regression Coefficients Plot**
st.subheader("Regression Coefficients by Cluster")
fig_coeff = px.bar(regression_df.melt(id_vars="Cluster", var_name="Metric", value_name="Value"),
                   x="Cluster", y="Value", color="Metric", barmode="group",
                   title="Regression Coefficients by Cluster")
st.plotly_chart(fig_coeff)

st.write("### **Insights**")
st.write("""
- Different clusters show distinct behaviors of electricity price concerning temperature and wind.
- The Wind coefficient is relatively small, indicating wind has a limited effect on electricity prices.
- Some clusters may have a negative temperature coefficient, meaning higher temperatures reduce prices.
""")
