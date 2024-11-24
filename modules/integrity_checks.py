import pandas as pd
from colorama import Fore, Style

def format_message(message, color=Fore.GREEN):
    """
    Utility function to format and print messages with color.
    """
    print(color + message + Style.RESET_ALL)

def format_number(num):
    return f"{num:,}"

def report_missing_values(missing_values, dataset_name):
    """
    Report missing values in a dataset if they exist.
    """
    if missing_values.sum() > 0:
        format_message(f"\nMissing Values Summary for {dataset_name}:", Fore.YELLOW)
        print(missing_values[missing_values > 0].to_string())
    else:
        format_message(f"\nNo missing values found in {dataset_name}.")

def check_missing_values(df, dataset_name):
    """
    Function to check for missing values in a dataset and report them.
    """
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(Fore.YELLOW + f"\nMissing Values Summary for {dataset_name}:" + Style.RESET_ALL)
        print(missing_values[missing_values > 0].to_string())
    else:
        print(Fore.GREEN + f"\nNo missing values found in {dataset_name}." + Style.RESET_ALL)

def remove_negative_unit_price(df, dataset_name):
    """
    Function to remove rows with negative unit prices from the dataset.
    """
    negative_unit_price_rows = df[df['UnitPrice'] < 0]
    if not negative_unit_price_rows.empty:
        print(Fore.YELLOW + f"\n{len(negative_unit_price_rows)} rows removed from {dataset_name} due to negative UnitPrice (possible discounts)." + Style.RESET_ALL)
        df = df[df['UnitPrice'] >= 0]
    else:
        print(Fore.GREEN + f"\nNo rows with negative UnitPrice in {dataset_name}." + Style.RESET_ALL)
    
    return df

def handle_missing_nps(df):
    """
    Handle missing values in the 'nps' field.
    """
    try:
        df['nps'] = pd.to_numeric(df['nps'].replace(' ', pd.NA), errors='coerce')
        missing_nps = df['nps'].isna().sum()

        if missing_nps > 0:
            format_message(f"Warning: 'nps' column has {format_number(missing_nps)} missing values.", Fore.RED)
        else:
            format_message("No missing values in the 'nps' column.", Fore.GREEN)
        
        return df
    except Exception as e:
        format_message(f"Error in handling missing 'nps': {str(e)}", Fore.RED)
        return df

def check_for_duplicates(df, dataset_name):
    """
    Function to check for duplicate rows in the dataset and report them.
    """
    initial_shape = df.shape
    duplicates = df.duplicated(keep=False)
    if duplicates.sum() > 0:
        print(Fore.RED + f"\n{dataset_name} - Duplicate Rows Found: {duplicates.sum()}." + Style.RESET_ALL)
        print(Fore.YELLOW + "Here are the first few examples:" + Style.RESET_ALL)
        print(df[duplicates].head().to_string(index=False))

        # Clarify how duplicates are handled
        print(Fore.CYAN + "\nNote: One occurrence of each duplicate group will be kept." + Style.RESET_ALL)
    else:
        print(Fore.GREEN + f"No duplicate rows found in {dataset_name}." + Style.RESET_ALL)
    
    df = df.drop_duplicates()
    rows_dropped = initial_shape[0] - df.shape[0]
    print(Fore.GREEN + f"Duplicate rows removed: {rows_dropped}. Remaining rows: {df.shape[0]}." + Style.RESET_ALL)
    
    return df

def perform_integrity_checks(online_retail_data, campaign_response_data):
    """
    This function performs all integrity checks for both datasets.
    """
    # Check for missing values in both datasets
    check_missing_values(online_retail_data, 'Online Retail Data')
    check_missing_values(campaign_response_data, 'Campaign Response Data')

    # Remove rows with negative UnitPrice for Online Retail Data
    online_retail_data = remove_negative_unit_price(online_retail_data, 'Online Retail Data')

    # Check for duplicates in both datasets
    online_retail_data = check_for_duplicates(online_retail_data, 'Online Retail Data')
    campaign_response_data = check_for_duplicates(campaign_response_data, 'Campaign Response Data')

    print(Fore.GREEN + f"\nFinal row count for Online Retail Data: {online_retail_data.shape[0]}." + Style.RESET_ALL)
    print(Fore.GREEN + f"Final row count for Campaign Response Data: {campaign_response_data.shape[0]}." + Style.RESET_ALL)

    return online_retail_data, campaign_response_data
