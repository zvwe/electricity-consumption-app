import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
from datetime import datetime
import os
import streamlit as st

#Run no command line : streamlit run elec.py

def load_and_clean_data(df):
    # Delete column 'A'
    if 'A' in df.columns:
        df = df.drop(columns=['A'])

    # Rename columns
    df = df.rename(columns={
        'תאריך': 'Date',
        'מועד תחילת הפעימה': 'DayTime',
        'צריכה בקוט"ש': 'Consumption'
    })

    # Convert 'Date' and 'DayTime' columns to a single datetime column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['DayTime'], format='%d/%m/%Y %H:%M', errors='coerce')

    # Remove rows with invalid dates
    df = df.dropna(subset=['Datetime'])

    # Remove all rows where the month is May
    df = df[df['Datetime'].dt.month != 5]

    # Check and handle missing values (assuming 0 for missing consumption)
    df['Consumption'] = df['Consumption'].fillna(0)

    return df

def plot_month_over_month_consumption(df):
    # Calculate monthly consumption
    df['Month'] = df['Datetime'].dt.to_period('M')
    monthly_consumption = df.groupby('Month')['Consumption'].sum()

    # Plot
    fig, ax = plt.subplots()
    monthly_consumption.plot(kind='bar', title='Month-over-Month Electricity Consumption', ax=ax)
    ax.set_ylabel('Total Consumption (kWh)')
    ax.set_xlabel('Month')
    plt.tight_layout()
    st.pyplot(fig)

def plot_month_over_month_per_day_consumption(df):
    # Define the correct order for days of the week, starting with Sunday
    days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    # Convert 'Weekday' to a categorical type with the specified order
    df['Weekday'] = pd.Categorical(df['Datetime'].dt.day_name(), categories=days_order, ordered=True)
    df['Month'] = df['Datetime'].dt.to_period('M')

    avg_weekly_consumption = df.groupby(['Month', 'Weekday'], observed=False)['Consumption'].sum().unstack()

    # Plot
    fig, ax = plt.subplots()
    avg_weekly_consumption.plot(kind='bar', title='Total Consumption by Day (Month-over-Month)', ax=ax)
    ax.set_ylabel('Total Consumption')
    plt.xticks(rotation=30)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=4)
    plt.tight_layout()
    st.pyplot(fig)

def plot_day_hour_consumption(df):
    df['Hour'] = df['Datetime'].dt.hour
    df['Month'] = df['Datetime'].dt.to_period('M')  # Use period to group by month

    # Sum the consumption per hour per day
    hourly_sum = df.groupby(['Month', 'Date', 'Hour'])['Consumption'].sum()

    # Calculate the average hourly consumption per month
    average_hourly_consumption = hourly_sum.groupby(['Month', 'Hour']).mean().unstack()

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(average_hourly_consumption.index)))

    for i, month in enumerate(average_hourly_consumption.index):
        ax.plot(average_hourly_consumption.columns, average_hourly_consumption.loc[month], label=f'Month {month}', color=colors[i])

    ax.set_title('Average Hourly Consumption (Month-over-Month)')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Average Consumption (kWh)')
    ax.legend(title='Month')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

def detect_anomalies(df, threshold=3):
    df['HourStart'] = df['Datetime'].dt.floor('h')

    # Group by day of the week and full hour
    df['DayOfWeek'] = df['HourStart'].dt.day_name()
    df['Hour'] = df['HourStart'].dt.hour

    # Aggregate consumption by full hour
    hourly_consumption = df.groupby(['DayOfWeek', 'HourStart', 'Hour'])['Consumption'].sum().reset_index()

    # Calculate mean and standard deviation for each day of the week and hour
    day_hour_stats = hourly_consumption.groupby(['DayOfWeek', 'Hour']).agg({
        'Consumption': ['mean', 'std']
    }).reset_index()
    day_hour_stats.columns = ['DayOfWeek', 'Hour', 'MeanConsumption', 'StdConsumption']

    # Merge the calculated mean and std back with the original dataframe
    hourly_consumption = hourly_consumption.merge(day_hour_stats, on=['DayOfWeek', 'Hour'])

    # Calculate the Z-score to detect anomalies
    hourly_consumption['ZScore'] = np.abs((hourly_consumption['Consumption'] - hourly_consumption['MeanConsumption']) / hourly_consumption['StdConsumption'])

    # Identify anomalies where the Z-score is above the threshold
    anomalies = hourly_consumption[hourly_consumption['ZScore'] > threshold]

    return anomalies

def main():
    st.title("Electricity Consumption Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Please upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded file
        try:
            df = pd.read_csv(uploaded_file, skiprows=list(range(10)) + [11])
            df = load_and_clean_data(df)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        st.success("Data loaded successfully!")

        
        # Plots
        st.subheader("Month-over-Month Electricity Consumption")
        plot_month_over_month_consumption(df)

        st.subheader("Total Consumption by Day (Month-over-Month)")
        plot_month_over_month_per_day_consumption(df)

        st.subheader("Average Hourly Consumption (Month-over-Month)")
        plot_day_hour_consumption(df)

        # Detect anomalies
        anomalies = detect_anomalies(df)

        if not anomalies.empty:
            st.subheader("Anomalies Detected")
            st.dataframe(anomalies[['HourStart', 'DayOfWeek', 'Hour', 'Consumption', 'MeanConsumption', 'StdConsumption', 'ZScore']].round(2))
        else:
            st.subheader("No anomalies detected.")

    else:
        st.info("Awaiting CSV file upload.")

def run():
    main()

if __name__ == "__main__":
    run()
