import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Data
df = pd.read_csv("data/fmi_weather_and_price.csv", parse_dates=['Time'])


# Streamlit App
st.title("Weather Condition and Electricity Pricing Analysis")
st.write("""In this analysis we are trying to investigate the correlation between weather and the electricity spot price in Finand.
         Below graph display the variation of the price and temperature over the period of time for a two year starting from 2020.""")

# Create a subplot with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add Price to the primary y-axis
fig.add_trace(
    go.Scatter(x=df["Time"], y=df["Price"], mode="lines", line=dict(color="blue"), name="Price"),
    secondary_y=False,
)

# Add Temperature to the secondary y-axis
fig.add_trace(
    go.Scatter(x=df["Time"], y=df["Temp"], mode="lines", line=dict(color="brown"), name="Temperature"),
    secondary_y=True,
)

# Update axis titles
fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Price", secondary_y=False)
fig.update_yaxes(title_text="Temperature", secondary_y=True)

# Update layout
fig.update_layout(title="Price and Temperature Over Time")

# Show the plot in Streamlit
st.plotly_chart(fig)

st.write("""According to the plot, the price has been increased over the period of time.
         However, it is not very clear the variation of the price for different temperature value. Thus we are going to analyze the correlation between each parameter.""")

st.write("When find the correlation values for each other, we can analyse the pair-wise linear relationship. ")
# Resample data to daily average
df_daily = df.resample('D', on='Time').mean().reset_index()

# Compute correlation matrix
corr_matrix = df_daily.corr()

# Display correlation matrix in Streamlit
st.write("###### Correlation Matrix for daily mean values")
st.write(""" Below table illustrates the correlation between average out the daily spot price and the weather conditions """)
st.dataframe(corr_matrix)  # Use st.dataframe() for an interactive table

st.write("Based on the calculated values, the correlation between electricity price, wind speed, and temperature appears to be relatively low. As mentioned earlier, electricity prices tend to increase over time, with a correlation value of 0.63. However, the correlation results indicate that prices decrease with higher wind speeds but increase with rising temperatures. This finding appears contradictory to the general expectation, as increased wind power generation typically lowers electricity prices, while higher temperatures should reduce heating demand and potentially decrease prices.")


st.write("According to economic theory, electricity prices should generally decrease as temperatures rise. This is because higher temperatures typically reduce heating demand, leading to lower electricity consumption. In Finland, the summer months are characterized by long daylight hours, with the sun shining almost continuously in some regions. This extended daylight reduces the need for artificial lighting and heating, potentially lowering electricity demand. However, other factors such as increased cooling demand, renewable energy generation, and market dynamics can also influence electricity prices during warmer periods.")
# Sidebar for user input
st.write("#### Linear Regression: Temperature vs. Price")
st.write("Before we conclude, we are going to analyse the relationship between price and temperature using linear regration method. By changing the setting in the side bar, you can analyse the linear regration variation for different presentage of test data")
st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, step=0.05, value=0.2)

# Load Data (Ensure df is available)
# df = pd.read_csv("your_data.csv")  # Uncomment and modify if needed

# Feature selection

X = df[['Temp']]
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"**Root Mean Square Error:** {rmse:.2f}")


# Generate regression line
x_range = pd.Series(sorted(df['Temp']))  # Ensure x-values are sorted
y_range = model.intercept_ + model.coef_[0] * x_range  # y = b0 + b1*x


# Adjust scatter plot opacity to make the line visible



fig2 = px.scatter(df, x='Temp', y='Price', trendline="ols", title='Temperature vs Price')
fig2.add_scatter(x=X_test['Temp'], y=y_pred, mode='lines', name='Regression Line')
fig2.update_traces(marker=dict(color='blue'))
fig2.update_traces(line=dict(color='red'))
st.plotly_chart(fig2, use_container_width=True)

df.corr()
results = px.get_trendline_results(fig2)
results.iloc[0].px_fit_results.summary()
# Get trendline results
#results = px.get_trendline_results(fig1)

# Display the regression summary in Streamlit
st.text(results.iloc[0].px_fit_results.summary())

st.write(""""When analyze the statistics in the given table, x1 is statistically significant (p < 0.05), meaning it has a real effect on y.
 However, the model explains only 2.2% of the variance in y, making it a poor predictor.""")


st.write( "### What matters most??")
st.write("In practice, we know that spot prices are influenced by both electricity consumption and production dynamics. Therefore, analyzing only the relationship between temperature and spot price does not fully capture the underlying dependencies. ")
st.write("#### 1.1. Time and seasonal effect on spot price for different weather conditions")
st.write("To gain better insights, we will first examine the correlation across different time slots and then different months. Instead of using a regression model, we will simply compute the correlation matrix to identify potential linear relationships between variables.")





# Define empty lists to store correlation values and time slots
time_slots_labels = []
price_temp_correlation = []
temp_wind_correlation = []




# Set "Time" as the index for time-based filtering
df.set_index("Time", inplace=True)

# Select relevant columns
#df_selected = df[["Wind", "Temp", "Price"]]
df_selected = df[["Temp", "Price"]].copy()  # Copying to avoid SettingWithCopyWarning
df_selected["Wind"] = df["Wind"] ** 3

# Define 2-hour time slots starting at 8:00 AM
time_slots = pd.date_range(start="08:00", periods=13, freq="2H").time

# Dictionary to store correlation results
correlation_results = {}

# Loop through each time slot and calculate correlations
for i in range(len(time_slots) - 1):
    start_time = time_slots[i]
    end_time = time_slots[i + 1]
    
    # Filter data for the time slot
    slot_data = df_selected.between_time(start_time, end_time)
    
    # Compute correlations
    price_temp_corr = slot_data["Price"].corr(slot_data["Temp"])
    price_wind_corr = slot_data["Price"].corr(slot_data["Wind"])
    
    # Store the correlations
    correlation_results[f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"] = {
        "Price-Temp Correlation": price_temp_corr,
        "Price-Wind Correlation": price_wind_corr
    }

# Convert results to DataFrame for better visualization
correlation_df = pd.DataFrame(correlation_results).T
st.write("##### Correlation Table for Price, Temperature and wind speed for different time of the day")
st.dataframe(correlation_df)
# Reset index to use time slots as a column
correlation_df = correlation_df.reset_index().rename(columns={"index": "Time Slot"})
st.write("""Above table illustrates the correaltion between price and temperature and price and wind speed for different time slots.
         Visualizing table data into a graph will help to analyze the values clearly. The minimum correaltion value is 0.03 and maximum is 1.92
         for price and tempearture. However, wind- price correlation is still low.""")
# Plot using Plotly
fig = px.line(
    correlation_df,
    x="Time Slot",
    y=["Price-Temp Correlation", "Price-Wind Correlation"],
    markers=True,
    labels={"value": "Correlation", "variable": "Correlation Type"},
    title="Price-Temp and Price-Wind Correlation Over different time slots"
)

# Customize layout
fig.update_layout(
    xaxis_title="Time Slot",
    yaxis_title="Correlation",
    xaxis_tickangle=-45,
    legend_title="Correlation Type"
)

# Show the plot in Streamlit

st.plotly_chart(fig)

st.write("")
# Define empty lists to store correlation values
month_list = []
time_slots_labels = []
price_temp_correlation = []
price_wind_correlation = []

# Loop through each month
for month in range(1, 13):  # 1 = January, 12 = December
    df_month = df_selected[df_selected.index.month == month]  # Filter by month
    
    # Loop through each time slot
    for i in range(len(time_slots) - 1):
        start_time = time_slots[i]
        end_time = time_slots[i + 1]

        # Store labels
        month_list.append(month)
        time_slots_labels.append(f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")

        # Filter data for the time slot
        slot_data = df_month.between_time(start_time, end_time)

        # Compute correlations
        price_temp_corr = slot_data["Price"].corr(slot_data["Temp"])
        price_wind_corr = slot_data["Price"].corr(slot_data["Wind"])

        # Append results
        price_temp_correlation.append(price_temp_corr)
        price_wind_correlation.append(price_wind_corr)

# Create a DataFrame
seasonal_correlation_df = pd.DataFrame({
    "Month": month_list,
    "Time Slot": time_slots_labels,
    "Price-Temp Correlation": price_temp_correlation,
    "Price-Wind Correlation": price_wind_correlation
})

# Streamlit App
st.write(" #### Seasonal Correlation Analysis")

# Show the correlation data
st.write(" ##### corrrelation table for different months")
st.dataframe(seasonal_correlation_df)

# Interactive Plotly Heatmap
st.write(" #### Monthly-Time Slot Correlation Heatmap")

# Pivot for heatmap (Price-Temp)
heatmap_df_temp = seasonal_correlation_df.pivot(index="Time Slot", columns="Month", values="Price-Temp Correlation")


# Ensure the pivot table includes all time slots and months, filling missing values with NaN
heatmap_df_temp = seasonal_correlation_df.pivot(index="Time Slot", columns="Month", values="Price-Temp Correlation").fillna(0)
heatmap_df_wind = seasonal_correlation_df.pivot(index="Time Slot", columns="Month", values="Price-Wind Correlation").fillna(0)

# Ensure y-axis (Time Slot) matches heatmap rows
time_slots_sorted = sorted(seasonal_correlation_df["Time Slot"].unique())  # Sort time slots

# Reindex to enforce consistency
heatmap_df_temp = heatmap_df_temp.reindex(time_slots_sorted)
heatmap_df_wind = heatmap_df_wind.reindex(time_slots_sorted)

# Ensure column order is consistent for months
heatmap_df_temp = heatmap_df_temp.reindex(columns=range(1, 13), fill_value=0)
heatmap_df_wind = heatmap_df_wind.reindex(columns=range(1, 13), fill_value=0)

# Plot updated heatmaps
fig_temp = px.imshow(
    heatmap_df_temp,
    labels={"color": "Price-Temp Correlation"},
    title="Price-Temp Correlation (Monthly-Time Slot)",
    x=[f" {i}" for i in range(1, 13)],  # Month labels
    y=time_slots_sorted,  # Sorted Time Slot labels
    color_continuous_scale="RdBu_r"
)

fig_wind = px.imshow(
    heatmap_df_wind,
    labels={"color": "Price-Wind Correlation"},
    title="Price-Wind Correlation (Monthly-Time Slot)",
    x=[f" {i}" for i in range(1, 13)],
    y=time_slots_sorted,
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(fig_temp)
st.plotly_chart(fig_wind)

import streamlit as st

st.subheader("Conclusion on Electricity Price Behavior in Finland Considering Temperature & Wind")

st.write("""
The relationship between **electricity price, temperature, and wind speed in Finland** follows  some seasonal patterns, mostly this can be happened due to the influence of **heating demand, cooling demand, daylight hours, and renewable energy production**:

####  Key Insights:
- **Temperature and wind both influence electricity prices, but their effects vary across seasons.**
- **In winter, colder temperatures increase prices, while higher wind speeds sometimes amplify price spikes.**
- **In summer, higher temperatures decrease prices, and stronger wind speeds contribute to lower costs.**
- **Wind power has a mixed effect—raising prices in January and May but lowering them in April and December.**
- **Electricity pricing in Finland is highly seasonal, correlation between weather and price are also seasonal.** """)

st.markdown('<p style="color:red;">Analyzing electricity consumption and production alongside weather factors would provide better insights. However, Fingrid data for electricity consumption and production in 2020-2022 is unavailable, making it difficult to clearly map the relationship between temperature, consumption, and production. On the second page, consumption plots are available for pattern comparison. While price trends have changed over time, the consumption pattern can still be observed.</p>', unsafe_allow_html=True)
st.markdown('<p style="color:blue; font-size:18px;"><b>Suggestions:</b></p>', unsafe_allow_html=True)
st.markdown('<p style="color:blue;"><b> Use graph neural network to find the correlation between price, weather, production and consumption will be more effective.</b></p>', unsafe_allow_html=True)


import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv("data/fmi_weather_and_price.csv", parse_dates=['Time'])
df["Month"] = df["Time"].dt.month  # Extract month

# Define seasons
season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
}
df["Season"] = df["Month"].map(season_map)

# Streamlit App
st.title("Seasonal Regression Analysis of Electricity Price")
st.write("### Impact of Temperature & Wind on Electricity Price")

# Function to perform regression and create Plotly figure
def plot_regression(df_season, x_feature, y_feature, season):
    X = df_season[[x_feature]].values.reshape(-1, 1)
    y = df_season[y_feature].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate regression line
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_range)
    
    fig = px.scatter(df_season, x=x_feature, y=y_feature, 
                     title=f"{y_feature} vs {x_feature} in {season}", 
                     labels={x_feature: x_feature, y_feature: y_feature}, 
                     trendline="ols")
    
    fig.add_trace(go.Scatter(x=X_range.flatten(), y=y_pred, mode='lines', 
                             name='Regression Line', line=dict(color='red')))
    return fig

# Loop through seasons
for season in ["Winter", "Spring", "Summer", "Autumn"]:
    df_season = df[df["Season"] == season]
    
    st.subheader(f"{season} Regression Analysis")
    
    fig_temp = plot_regression(df_season, "Temp", "Price", season)
    st.plotly_chart(fig_temp)
    
    fig_wind = plot_regression(df_season, "Wind", "Price", season)
    st.plotly_chart(fig_wind)
