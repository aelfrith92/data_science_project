import random
import string
from colorama import Fore, Style
import pandas as pd

def generate_random_identifier(existing_ids, length=8):
    """
    Generate a random alphanumeric identifier and ensure it doesn't exist in the dataset.
    """
    while True:
        random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        if random_id not in existing_ids:
            return random_id

def handle_missing_values(df, column_name):
    """
    Provide user with options to handle missing values based on the data type of the column.
    """
    missing_count = df[column_name].isnull().sum() + df[column_name].isin(['', ' ']).sum()
    if missing_count > 0:
        print(f"\n{Fore.YELLOW}Column '{column_name}' has {missing_count} missing or invalid values.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}How would you like to handle the missing values?{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}1. Delete rows with missing values.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}2. Fill missing values with a default (based on data type).{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}3. Keep the data unvaried (do nothing).{Style.RESET_ALL}")
        
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            # Option to delete rows with missing values
            df = df.dropna(subset=[column_name])
            print(f"{Fore.GREEN}{missing_count} rows removed from '{column_name}' due to missing values.{Style.RESET_ALL}")
        
        elif choice == '2':
            # Fill missing based on data type
            if df[column_name].dtype == 'object':
                # For strings, fill with 'Unknown-x'
                existing_unknowns = df[column_name].str.contains('Unknown').sum()
                df[column_name] = df[column_name].fillna(f'Unknown-{existing_unknowns + 1}')
                print(f"{Fore.GREEN}Filled missing values in '{column_name}' with 'Unknown'.{Style.RESET_ALL}")
            
            elif df[column_name].dtype in ['float64', 'int64']:
                # For numerical columns, fill with mode
                mode_value = df[column_name].mode()[0]
                df[column_name] = df[column_name].fillna(mode_value)
                print(f"{Fore.GREEN}Filled missing values in '{column_name}' with the mode value: {mode_value}.{Style.RESET_ALL}")
            
            elif pd.api.types.is_categorical_dtype(df[column_name]):
                # For categorical columns, fill with the mode
                mode_value = df[column_name].mode()[0]
                df[column_name] = df[column_name].fillna(mode_value)
                print(f"{Fore.GREEN}Filled missing values in '{column_name}' with the mode value: {mode_value}.{Style.RESET_ALL}")
        
        elif choice == '3':
            # Option to keep the data unchanged
            print(f"{Fore.GREEN}No changes made to the column '{column_name}'. Data remains unvaried.{Style.RESET_ALL}")
        
        else:
            print(Fore.RED + "Invalid choice, please select 1, 2, or 3." + Style.RESET_ALL)
    
    return df

def handle_missing_identifiers(df, identifier_column):
    """
    Handle missing or invalid values in identifier columns (CustomerID, InvoiceNo, StockCode)
    by giving the user the option to fill missing values with random unique identifiers or delete the rows.
    """
    missing_count = df[identifier_column].isnull().sum() + df[identifier_column].isin(['', ' ']).sum()

    if missing_count > 0:
        print(f"\n{Fore.YELLOW}{missing_count} missing or invalid values found in '{identifier_column}'." + Style.RESET_ALL)
        print(f"{Fore.CYAN}How would you like to handle these values?{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}1. Fill missing values with random unique identifiers.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}2. Delete rows with missing or invalid values.{Style.RESET_ALL}")
        choice = input("Enter your choice (1/2): ")

        if choice == '1':
            # Fill with random unique identifiers
            existing_ids = df[identifier_column].dropna().unique()
            df.loc[df[identifier_column].isnull() | df[identifier_column].isin(['', ' ']), identifier_column] = df.apply(
                lambda _: generate_random_identifier(existing_ids), axis=1)
            print(f"{Fore.GREEN}Missing values in '{identifier_column}' have been filled with random unique identifiers.{Style.RESET_ALL}")
        
        elif choice == '2':
            # Delete rows with missing or invalid values
            df = df.dropna(subset=[identifier_column])
            print(f"{Fore.GREEN}{missing_count} rows with missing or invalid values in '{identifier_column}' have been deleted.{Style.RESET_ALL}")
        
        else:
            print(Fore.RED + "Invalid choice, please select either 1 or 2." + Style.RESET_ALL)
    
    return df

def process_missing_data(df):
    """
    Loop through the columns of the dataframe and handle missing values based on data type.
    """
    for column in df.columns:
        if column in ['CustomerID', 'InvoiceNo', 'StockCode']:
            df = handle_missing_identifiers(df, column)
        else:
            df = handle_missing_values(df, column)
    
    return df