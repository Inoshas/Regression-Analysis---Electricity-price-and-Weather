import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# Load the Excel file for total consumption data
df_consumption = pd.read_excel('data/consumptions.xlsx')

# Create a line plot for Total Electricity Consumption
st.markdown("### Total electricity consumption in Finland")
st.write("The below graph illustrates the total electricity consumption for different times of the year. FIngrid has data from August 2023 onwards.  It shows that during winter, power consumption increases, as expected, due to the dark and cold weather. (Please note that this is the same graph I used in assignment 2).")

line_plot = px.line(df_consumption, x="endTime", y="Total_electricity_consumption",
                    title="Total Electricity Consumption in MWh/h",
                    labels={"endTime": "Time", "Total_electricity_consumption": "Electricity Consumption (MWh/h)"})
st.plotly_chart(line_plot)



# Convert 'endTime' to datetime format
df_consumption["endTime"] = pd.to_datetime(df_consumption["endTime"])

# Extract month and time slot (2-hour intervals)
df_consumption["Month"] = df_consumption["endTime"].dt.month
df_consumption["Hour"] = df_consumption["endTime"].dt.hour

# Define time slots (grouping hours into 2-hour slots)
df_consumption["Time Slot"] = (df_consumption["Hour"] // 2) * 2

# Group by Month and Time Slot and calculate average consumption
heatmap_data = df_consumption.groupby(["Month", "Time Slot"])["Total_electricity_consumption"].mean().reset_index()

# Pivot the data to create a matrix for the heatmap
heatmap_pivot = heatmap_data.pivot(index="Time Slot", columns="Month", values="Total_electricity_consumption")

# Create the heatmap using Plotly
fig = px.imshow(heatmap_pivot,
                labels=dict(x="Month", y="Time Slot (Hours)", color="Avg Consumption (MWh/h)"),
                x=heatmap_pivot.columns, 
                y=heatmap_pivot.index,
                color_continuous_scale="RdBu_r",
                title="Electricity Consumption Heatmap by Month and Time Slot")

# Increase figure size and improve readability
fig.update_layout(
    width=400,  # Wider figure
    height=600,  # Shorter figure
    font=dict(size=14),
    xaxis=dict(scaleanchor=None),  # Removes auto-scaling of width/height
    yaxis=dict(scaleanchor=None)  
)


# Display in Streamlit
st.markdown("###  Electricity Consumption Heatmap")
st.write("This heatmap illustrates how electricity consumption changes across different months and time slots. **Darker colors indicate lower consumption, while brighter colors show peak demand periods.**")
st.plotly_chart(fig)


