# Streamlit App with Regression Analysis

## Overview
This Streamlit application analyzes the relationship between weather conditions and electricity prices in Finland. It uses data visualization, correlation analysis, and linear regression to uncover insights into how weather factors impact electricity pricing. Developed  as part of an energy data analysis project. 

## Features
- Data Upload: Allows users to upload datasets containing weather conditions and electricity prices.

- Data Preprocessing: Cleans and processes data, handling missing values where necessary.

- Correlation Analysis: Computes correlation coefficients between different weather variables and electricity prices.

- Linear Regression Models: Fits regression models to examine the impact of weather variables on electricity prices.

- Visualization: Generates plots, including scatter plots and regression lines, for better data interpretation.

- Conclusion Summary: Provides key insights derived from the analysis.

## Installation
To run this Streamlit app, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/Inoshas/Regression-Analysis---Electricity-price-and-Weather/tree/main
   cd <repository_name>
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```sh
   pip install streamlit pandas numpy matplotlib seaborn sklearn
   ```

## Usage
Run the Streamlit app using the following command:
   ```sh
   streamlit run main.py
   ```

## File Structure
```
project_root/
│── main.py               # Main Streamlit app
│── pages/               # Other pages in streamlit app
│── data/                # (Optional) Data directory
│── README.md            # This file
```



