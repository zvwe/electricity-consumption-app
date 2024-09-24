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
    avg_weekly_consumption.plot(kind='bar', title='Total Consumption by Day (Month-over-Month)')
    plt.ylabel('Average Consumption')
    plt.xlabel('Month')
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

def main():
    # Ask the user to input the file path
    file_path = input("Please enter the path to your CSV file (e.g., meter_Aug2024.csv): ")

    # Check if the file exists
    if not os.path.exists(file_path):
       print(f"Error: The file '{file_path}' does not exist.")
       return

    #file_path = 'meter_spt2024.csv'  # Adjust to your actual file path

        # Generate a timestamped filename for the PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputPlots_{timestamp}.pdf"

    df = load_and_clean_data(file_path)
    
    # Create a PDF file to save the plots
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
  

if __name__ == "__main__":
    main()
