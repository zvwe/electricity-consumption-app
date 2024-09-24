import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
from datetime import datetime
import os
import streamlit as st
from PIL import Image
import io
import requests

# Run without command line: streamlit run elec.py

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

def calculate_bezeq_packages(df):
    # Calculate base consumption
    df['BaseUsage'] = df['Consumption']
    
    # Package 1: 15% discount for Sunday to Thursday between 07:00 to 17:00
    df['Package1'] = df['BaseUsage'].where(
        ~((df['Datetime'].dt.dayofweek.isin([6, 0, 1, 2, 3])) &  # Sunday to Thursday
          (df['Datetime'].dt.hour.between(7, 16))),  # 07:00 to 17:00
        df['BaseUsage'] * 0.85
    )
    
    # Package 2: Constant 7% discount
    df['Package2'] = df['BaseUsage'] * 0.93
    
    # Package 3: 20% discount for Sunday to Thursday between 23:00 to 07:00
    df['Package3'] = df['BaseUsage'].where(
        ~((df['Datetime'].dt.dayofweek.isin([6, 0, 1, 2, 3])) &  # Sunday to Thursday
          ((df['Datetime'].dt.hour >= 23) | (df['Datetime'].dt.hour < 7))),  # 23:00 to 07:00
        df['BaseUsage'] * 0.80
    )
    
    # Calculate total usage for each package
    total_usage = {
        'Base Usage': df['BaseUsage'].sum(),
        'Package 1': df['Package1'].sum(),
        'Package 2': df['Package2'].sum(),
        'Package 3': df['Package3'].sum()
    }
    
    return total_usage

def display_bezeq_comparison(df):
    st.subheader("Bezeq Electricity Package Comparison")

    # Add Bezeq logo
    logo_path = "assets/logos/bezeq.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=200)
    else:
        st.warning("Bezeq logo not found. Please check the file path.")

    # Calculate package usage
    bezeq_usage = calculate_bezeq_packages(df)

    # Create three columns for package display
    col1, col2, col3 = st.columns(3)

    # Calculate percentage savings for each package
    base_usage = bezeq_usage['Base Usage']
    savings = {k: (base_usage - v) / base_usage * 100 for k, v in bezeq_usage.items() if k != 'Base Usage'}

    # Display package information in cards
    with col1:
        st.markdown("### Package 1")
        st.markdown("15% discount for Sunday to Thursday, 07:00-17:00")
        st.metric("Savings", f"{savings['Package 1']:.2f}%")

    with col2:
        st.markdown("### Package 2")
        st.markdown("Constant 7% discount")
        st.metric("Savings", f"{savings['Package 2']:.2f}%")

    with col3:
        st.markdown("### Package 3")
        st.markdown("20% discount for Sunday to Thursday, 23:00-07:00")
        st.metric("Savings", f"{savings['Package 3']:.2f}%")

    # Find the best package
    best_package = max(savings, key=savings.get)
    best_savings = savings[best_package]

    # Display results with improved styling
    st.markdown("---")
    st.markdown(f"### 🏆 Best Package: {best_package}")
    st.markdown(f"**Potential Savings:** {best_savings:.2f}%")

    # Visualize savings with a bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    packages = list(savings.keys())
    savings_values = list(savings.values())
    colors = ['#1f77b4' if package != best_package else '#2ca02c' for package in packages]
    
    ax.bar(packages, savings_values, color=colors)
    ax.set_ylabel('Savings (%)')
    ax.set_title('Bezeq Package Comparison - Potential Savings')
    plt.xticks(rotation=45)
    
    for i, saving in enumerate(savings_values):
        ax.text(i, saving, f'{saving:.2f}%', ha='center', va='bottom')
    
    st.pyplot(fig)

    # Display usage patterns
    st.subheader("Your Electricity Usage Patterns")

    # Calculate percentage of usage during discount hours for Package 1 and 3
    package1_discount_usage = df[
        (df['Datetime'].dt.dayofweek.isin([6, 0, 1, 2, 3])) &  # Sunday to Thursday
        (df['Datetime'].dt.hour.between(7, 16))  # 07:00 to 17:00
    ]['Consumption'].sum()

    package3_discount_usage = df[
        (df['Datetime'].dt.dayofweek.isin([6, 0, 1, 2, 3])) &  # Sunday to Thursday
        ((df['Datetime'].dt.hour >= 23) | (df['Datetime'].dt.hour < 7))  # 23:00 to 07:00
    ]['Consumption'].sum()

    total_usage = df['Consumption'].sum()

    package1_discount_percentage = (package1_discount_usage / total_usage) * 100
    package3_discount_percentage = (package3_discount_usage / total_usage) * 100

    st.markdown(f"- {package1_discount_percentage:.2f}% of your usage is during Package 1 discount hours (Sun-Thu, 07:00-17:00)")
    st.markdown(f"- {package3_discount_percentage:.2f}% of your usage is during Package 3 discount hours (Sun-Thu, 23:00-07:00)")

    # Provide recommendations based on usage patterns
    st.subheader("Recommendations")
    if best_package == 'Package 1':
        st.markdown("You could benefit from shifting more of your usage to Sunday-Thursday, 07:00-17:00 to maximize your savings with Package 1.")
    elif best_package == 'Package 3':
        st.markdown("You could benefit from shifting more of your usage to Sunday-Thursday, 23:00-07:00 to maximize your savings with Package 3.")
    else:
        st.markdown("Your usage is fairly consistent throughout the day. Package 2 with its constant discount might be the most convenient for you.")

def calculate_cellcom_packages(df):
    # Calculate base consumption
    df['BaseUsage'] = df['Consumption']
    
    # Package 1: 18% discount all days, between 14:00 to 20:00
    df['Package1_Cellcom'] = df['BaseUsage'].where(
        ~df['Datetime'].dt.hour.between(14, 19),  # 14:00 to 20:00 (upper bound exclusive)
        df['BaseUsage'] * 0.82
    )
    
    # Package 2: Constant 5% discount
    df['Package2_Cellcom'] = df['BaseUsage'] * 0.95
    
    # Package 3: 15% discount from Sunday to Thursday between 07:00 to 17:00
    df['Package3_Cellcom'] = df['BaseUsage'].where(
        ~((df['Datetime'].dt.dayofweek.isin([6, 0, 1, 2, 3])) &  # Sunday to Thursday
          (df['Datetime'].dt.hour.between(7, 16))),  # 07:00 to 17:00
        df['BaseUsage'] * 0.85
    )
    
    # Package 4: 20% discount all days between 23:00-07:00
    df['Package4_Cellcom'] = df['BaseUsage'].where(
        ~(
            (df['Datetime'].dt.hour >= 23) | (df['Datetime'].dt.hour < 7)
        ),  # 23:00 to 07:00
        df['BaseUsage'] * 0.80
    )
    
    # Calculate total usage for each package
    total_usage = {
        'Base Usage': df['BaseUsage'].sum(),
        'Package 1': df['Package1_Cellcom'].sum(),
        'Package 2': df['Package2_Cellcom'].sum(),
        'Package 3': df['Package3_Cellcom'].sum(),
        'Package 4': df['Package4_Cellcom'].sum()
    }
    
    return total_usage

def display_cellcom_comparison(df):
    st.subheader("Cellcom Electricity Package Comparison")

    # Add Cellcom logo
    logo_path = "assets/logos/cellcom.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=200)
    else:
        st.warning("Cellcom logo not found. Please check the file path.")

    # Calculate package usage
    cellcom_usage = calculate_cellcom_packages(df)

    # Create four columns for package display
    col1, col2, col3, col4 = st.columns(4)

    # Calculate percentage savings for each package
    base_usage = cellcom_usage['Base Usage']
    savings = {k: (base_usage - v) / base_usage * 100 for k, v in cellcom_usage.items() if k != 'Base Usage'}

    # Display package information in cards
    with col1:
        st.markdown("### Package 1")
        st.markdown("18% discount all days, 14:00-20:00")
        st.metric("Savings", f"{savings['Package 1']:.2f}%")

    with col2:
        st.markdown("### Package 2")
        st.markdown("Constant 5% discount")
        st.metric("Savings", f"{savings['Package 2']:.2f}%")

    with col3:
        st.markdown("### Package 3")
        st.markdown("15% discount Sun-Thu, 07:00-17:00")
        st.metric("Savings", f"{savings['Package 3']:.2f}%")

    with col4:
        st.markdown("### Package 4")
        st.markdown("20% discount all days, 23:00-07:00")
        st.metric("Savings", f"{savings['Package 4']:.2f}%")

    # Find the best package
    best_package = max(savings, key=savings.get)
    best_savings = savings[best_package]

    # Display results with improved styling
    st.markdown("---")
    st.markdown(f"### 🏆 Best Cellcom Package: {best_package}")
    st.markdown(f"**Potential Savings:** {best_savings:.2f}%")

    # Visualize savings with a bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    packages = list(savings.keys())
    savings_values = list(savings.values())
    colors = ['#1f77b4' if package != best_package else '#2ca02c' for package in packages]
    
    ax.bar(packages, savings_values, color=colors)
    ax.set_ylabel('Savings (%)')
    ax.set_title('Cellcom Package Comparison - Potential Savings')
    plt.xticks(rotation=45)
    
    for i, saving in enumerate(savings_values):
        ax.text(i, saving, f'{saving:.2f}%', ha='center', va='bottom')
    
    st.pyplot(fig)

    # Display usage patterns
    st.subheader("Your Electricity Usage Patterns with Cellcom Packages")

    # Calculate percentage of usage during discount hours for each package
    package1_discount_usage = df[
        df['Datetime'].dt.hour.between(14, 19)  # 14:00 to 20:00
    ]['Consumption'].sum()
    
    package3_discount_usage = df[
        (df['Datetime'].dt.dayofweek.isin([6, 0, 1, 2, 3])) &  # Sunday to Thursday
        (df['Datetime'].dt.hour.between(7, 16))  # 07:00 to 17:00
    ]['Consumption'].sum()
    
    package4_discount_usage = df[
        (df['Datetime'].dt.hour >= 23) | (df['Datetime'].dt.hour < 7)  # 23:00 to 07:00
    ]['Consumption'].sum()
    
    total_usage = df['Consumption'].sum()
    
    package1_discount_percentage = (package1_discount_usage / total_usage) * 100
    package3_discount_percentage = (package3_discount_usage / total_usage) * 100
    package4_discount_percentage = (package4_discount_usage / total_usage) * 100
    
    st.markdown(f"- {package1_discount_percentage:.2f}% of your usage is during Package 1 discount hours (All days, 14:00-20:00)")
    st.markdown(f"- {package3_discount_percentage:.2f}% of your usage is during Package 3 discount hours (Sun-Thu, 07:00-17:00)")
    st.markdown(f"- {package4_discount_percentage:.2f}% of your usage is during Package 4 discount hours (All days, 23:00-07:00)")

    # Provide recommendations based on usage patterns
    st.subheader("Recommendations")
    if best_package == 'Package 1':
        st.markdown("You could benefit from shifting more of your usage to all days between 14:00-20:00 to maximize your savings with Package 1.")
    elif best_package == 'Package 3':
        st.markdown("You could benefit from shifting more of your usage to Sunday-Thursday, 07:00-17:00 to maximize your savings with Package 3.")
    elif best_package == 'Package 4':
        st.markdown("You could benefit from shifting more of your usage to all days between 23:00-07:00 to maximize your savings with Package 4.")
    else:
        st.markdown("Your usage is fairly consistent throughout the day. Package 2 with its constant discount might be the most convenient for you.")

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

        # Display Bezeq package comparison
        display_bezeq_comparison(df)

        # Display Cellcom package comparison
        display_cellcom_comparison(df)

    else:
        st.info("Awaiting CSV file upload.")

def run():
    main()

if __name__ == "__main__":
    run()
