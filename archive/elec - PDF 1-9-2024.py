import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
from datetime import datetime
import os

def load_and_clean_data(file_path):
    # Load the data, skipping the first 10 rows and the 12th row (which is the 11th row after skipping the first 10)
    df = pd.read_csv(file_path, skiprows=list(range(10)) + [11])
    
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
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['DayTime'], format='%d/%m/%Y %H:%M')
    
    # Remove all rows where the month is May
    df = df[df['Datetime'].dt.month != 5]
     
    # Check and handle missing values (assuming 0 for missing consumption)
    df['Consumption'] = df['Consumption'].fillna(0)
    
    # Additional cleaning steps can be added here (e.g., handling outliers)
    
    return df
#-----------------------------------------------------------

def plot_month_over_month_consumption(df):
    # Calculate monthly consumption
    df['Month'] = df['Datetime'].dt.to_period('M')
    monthly_consumption = df.groupby('Month')['Consumption'].sum()
    
    # Plot
    monthly_consumption.plot(kind='bar', title='Month-over-Month Electricity Consumption')
    plt.ylabel('Total Consumption(KWh')
    plt.xlabel('Month')
    plt.tight_layout()

#--------------------------------------------------------

def plot_month_over_month_per_day_consumption(df):
    # Calculate total weekly consumption by day
    # Define the correct order for days of the week, starting with Sunday
    days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    # Convert 'Weekday' to a categorical type with the specified order
    df['Weekday'] = pd.Categorical(df['Datetime'].dt.day_name(), categories=days_order, ordered=True)
    df['Month'] = df['Datetime'].dt.to_period('M')

    avg_weekly_consumption = df.groupby(['Month', 'Weekday'], observed=False)['Consumption'].sum().unstack()

    # Plot
    ax = avg_weekly_consumption.plot(kind='bar', title='Total Consumption by Day (Month-over-Month)')
    plt.ylabel('Average Consumption')

    # Rotate x-axis labels by 30 degrees
    plt.xticks(rotation=30)
    
    # Move the legend to the bottom of the plot
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=4)
    plt.tight_layout()
   

#--------------------------------------------------------------
def plot_day_hour_consumption(df):
    # Ensure 'Datetime' column is of datetime type
    if df['Datetime'].dtype != 'datetime64[ns]':
        df['Datetime'] = pd.to_datetime(df['Datetime'])

    df['Hour'] = df['Datetime'].dt.hour
    df['Month'] = df['Datetime'].dt.to_period('M')  # Use period to group by month
 
    # Sum the consumption per hour per day
    hourly_sum = df.groupby(['Month', 'Date', 'Hour'])['Consumption'].sum()
    
    # Calculate the average hourly consumption per month
    average_hourly_consumption = hourly_sum.groupby(['Month', 'Hour']).mean().unstack()
        
    plt.figure(figsize=(12, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(average_hourly_consumption.index)))  # Changed to 'plasma' colormap
    
    
    for i, month in enumerate(average_hourly_consumption.index):
        plt.plot(average_hourly_consumption.columns, average_hourly_consumption.loc[month], label=f'Month {month}', color=colors[i])

    plt.title('Average Hourly Consumption (Month-over-Month)')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Consumption (kWh)')
    plt.legend(title='Month')
    plt.grid(True)
    plt.tight_layout()
#---------------------------------------------------------------

def detect_anomalies(df, threshold=3):
    # Ensure 'Datetime' column is of datetime type
    if df['Datetime'].dtype != 'datetime64[ns]':
        df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Round down to the nearest hour to group by full hour
    df['HourStart'] = df['Datetime'].dt.floor('H')

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

    if not anomalies.empty:
        print("Anomalies detected on the following dates and hours:")
        for _, row in anomalies.iterrows():
            print(f"Date: {row['HourStart']}, Hour: {row['Hour']}: Consumption = {row['Consumption']:.2f} kWh "
                  f"(Day Mean = {row['MeanConsumption']:.2f} kWh, Day Std = {row['StdConsumption']:.2f} kWh, Z-Score = {row['ZScore']:.2f})")
    else:
        print("No anomalies detected.")

    return anomalies
#----------------------------------------------------------------

def main():
    # Ask the user to input the file path
    file_path = input("Please enter the path to your CSV file (e.g., meter_Aug2024.csv): ")

    # Check if the file exists
    if not os.path.exists(file_path):
       print(f"Error: The file '{file_path}' does not exist.")
       return

    # Generate a timestamped filename for the PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputPlots_{timestamp}.pdf"

    # Load and clean the data
    df = load_and_clean_data(file_path)

    # Detect anomalies
    anomalies = detect_anomalies(df)
    
    # Create a PDF file to save the plots and anomalies
    with PdfPages(output_file) as pdf:
        # Generate and save each plot
        plot_month_over_month_consumption(df)
        pdf.savefig()  # Save the current figure
        plt.close()    # Close the figure after saving

        plot_month_over_month_per_day_consumption(df)
        pdf.savefig()  # Save the current figure
        plt.close()    # Close the figure after saving

        plot_day_hour_consumption(df)
        pdf.savefig()  # Save the current figure
        plt.close()    # Close the figure after saving

        # Add a page for the anomalies
        if anomalies is not None and not anomalies.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('off')
            ax.axis('tight')

            # Convert the anomalies dataframe to a table and add to the plot
            table_data = anomalies[['HourStart', 'DayOfWeek', 'Hour', 'Consumption', 'MeanConsumption', 'StdConsumption', 'ZScore']].round(2)
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.2)

            ax.set_title('Detected Anomalies', fontsize=14)
            pdf.savefig()  # Save the table as a page in the PDF
            plt.close()
        else:
            # If no anomalies were found, add a simple text page
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No anomalies detected.', ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig()  # Save the text as a page in the PDF
            plt.close()

    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    main()
