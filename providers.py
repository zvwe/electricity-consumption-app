# providers.py

import yaml
import os
from PIL import Image
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def load_config(config_path='config/providers_config.yaml'):
    """
    Load provider configurations from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary containing provider configurations.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config['providers']
    except FileNotFoundError:
        st.error(f"Configuration file not found at {config_path}.")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file: {e}")
        return {}

def parse_conditions(df, conditions):
    """
    Parse the conditions from the configuration and return a boolean Series.

    Args:
        df (pd.DataFrame): The consumption DataFrame.
        conditions (dict): Conditions dictionary from the config.

    Returns:
        pd.Series: Boolean Series where True indicates the discount applies.
    """
    if not conditions:
        return pd.Series([True] * len(df), index=df.index)

    day_condition = True
    time_condition = True

    # Days of the week condition
    if 'days_of_week' in conditions and conditions['days_of_week']:
        days = conditions['days_of_week']
        day_condition = df['Datetime'].dt.dayofweek.isin(days)

    # Time condition
    if 'start_hour' in conditions and 'end_hour' in conditions:
        start = conditions['start_hour']
        end = conditions['end_hour']
        if start < end:
            # Simple range
            time_condition = df['Datetime'].dt.hour.between(start, end - 1)
        else:
            # Overnight range (e.g., 23:00-07:00)
            time_condition = (df['Datetime'].dt.hour >= start) | (df['Datetime'].dt.hour < end)

    # Combine day and time conditions
    combined_condition = day_condition & time_condition
    return combined_condition

def calculate_packages(df, provider_name, config):
    """
    Calculate package usages based on provider configuration.

    Args:
        df (pd.DataFrame): The consumption DataFrame.
        provider_name (str): Name of the provider.
        config (dict): Provider's package configuration.

    Returns:
        dict: Total usage per package including base usage.
    """
    # Initialize BaseUsage
    df['BaseUsage'] = df['Consumption']

    # Iterate over packages and calculate discounted usage
    for package_name, package_details in config['packages'].items():
        conditions = package_details.get('conditions', {})
        condition_series = parse_conditions(df, conditions)
        discount = package_details['discount']
        df[package_name] = df['BaseUsage'].where(
            ~condition_series,  # If condition is False, keep BaseUsage
            df['BaseUsage'] * discount  # Apply discount where condition is True
        )

    # Calculate total usage for each package
    total_usage = {'Base Usage': df['BaseUsage'].sum()}
    for package_name in config['packages']:
        total_usage[package_name] = df[package_name].sum()

    return total_usage

def display_package_comparison(df, provider_name, config):
    """
    Display package comparison for a given provider.

    Args:
        df (pd.DataFrame): The consumption DataFrame.
        provider_name (str): Name of the provider.
        config (dict): Provider's package configuration.
    """
    st.subheader(f"{provider_name} Electricity Package Comparison")

    # Add provider logo
    logo_path = config['logo_path']
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=200)
    else:
        st.warning(f"{provider_name} logo not found at {logo_path}. Please check the file path.")

    # Calculate package usage
    usage = calculate_packages(df, provider_name, config)

    # Create columns based on number of packages
    num_packages = len(config['packages'])
    cols = st.columns(num_packages)

    # Calculate percentage savings for each package
    base_usage = usage['Base Usage']
    savings = {k: (base_usage - v) / base_usage * 100 for k, v in usage.items() if k != 'Base Usage'}

    # Display package information in cards
    for idx, (package_name, package_details) in enumerate(config['packages'].items()):
        with cols[idx]:
            st.markdown(f"### {package_name}")
            st.markdown(package_details['description'])
            st.metric("Savings", f"{savings[package_name]:.2f}%")

    # Find the best package
    best_package = max(savings, key=savings.get)
    best_savings = savings[best_package]

    # Display results with improved styling
    st.markdown("---")
    st.markdown(f"### 🏆 Best {provider_name} Package: {best_package}")
    st.markdown(f"**Potential Savings:** {best_savings:.2f}%")

    # Visualize savings with a bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    packages = list(savings.keys())
    savings_values = list(savings.values())
    colors = ['#1f77b4' if package != best_package else '#2ca02c' for package in packages]

    ax.bar(packages, savings_values, color=colors)
    ax.set_ylabel('Savings (%)')
    ax.set_title(f'{provider_name} Package Comparison - Potential Savings')
    plt.xticks(rotation=45)

    for i, saving in enumerate(savings_values):
        ax.text(i, saving, f'{saving:.2f}%', ha='center', va='bottom')

    st.pyplot(fig)

    # Display usage patterns
    st.subheader(f"Your Electricity Usage Patterns with {provider_name} Packages")

    # Calculate percentage of usage during discount hours for applicable packages
    discount_usage = {}
    for package_name, package_details in config['packages'].items():
        if not package_details.get('conditions'):  # Skip constant discount packages
            continue
        conditions = package_details['conditions']
        condition_series = parse_conditions(df, conditions)
        discount_consumption = df[condition_series]['Consumption'].sum()
        total_consumption = df['Consumption'].sum()
        discount_percentage = (discount_consumption / total_consumption) * 100
        discount_usage[package_name] = discount_percentage
        desc = package_details['description']
        st.markdown(f"- {discount_percentage:.2f}% of your usage is during {package_name} discount hours ({desc})")

    # Provide recommendations based on usage patterns
    st.subheader("Recommendations")
    if best_package:
        # Identify if best package is a constant discount
        best_package_details = config['packages'][best_package]
        if not best_package_details.get('conditions'):
            st.markdown(f"Your usage is fairly consistent throughout the day. **{best_package}** with its constant discount might be the most convenient for you.")
        else:
            st.markdown(f"You could benefit from shifting more of your usage to the discount hours of **{best_package}** to maximize your savings.")
