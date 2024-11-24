import pandas as pd
from colorama import Fore, Style

def format_number(num):
    return f"{num:,}"

def describe_conversion(column_name, old_dtype, new_dtype):
    print(Fore.CYAN + f"Column '{column_name}': converted from {old_dtype} to {new_dtype}." + Style.RESET_ALL)

def convert_data_types(df):
    print(Fore.CYAN + "\nStarting data type conversions for Online Retail Data..." + Style.RESET_ALL)
    columns_converted = []

    # Convert CustomerID to string if it's numeric
    try:
        old_dtype = df['CustomerID'].dtype
        if old_dtype == 'int64' or old_dtype == 'float64':
            df['CustomerID'] = df['CustomerID'].astype(str)
            new_dtype = df['CustomerID'].dtype
            describe_conversion('CustomerID', old_dtype, new_dtype)
            columns_converted.append('CustomerID')
    except Exception as e:
        print(Fore.RED + f"Error converting 'CustomerID': {e}" + Style.RESET_ALL)

    # Convert Country to categorical
    try:
        old_dtype = df['Country'].dtype
        if old_dtype != 'category':
            df['Country'] = df['Country'].astype('category')
            new_dtype = df['Country'].dtype
            describe_conversion('Country', old_dtype, new_dtype)
            columns_converted.append('Country')
    except Exception as e:
        print(Fore.RED + f"Error converting 'Country': {e}" + Style.RESET_ALL)

    # Convert InvoiceDate to datetime
    try:
        old_dtype = df['InvoiceDate'].dtype
        if old_dtype != 'datetime64[ns]':
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%y %H:%M', errors='coerce')
            new_dtype = df['InvoiceDate'].dtype
            describe_conversion('InvoiceDate', old_dtype, new_dtype)
            columns_converted.append('InvoiceDate')
    except Exception as e:
        print(Fore.RED + f"Error converting 'InvoiceDate': {e}" + Style.RESET_ALL)

    if columns_converted:
        print(Fore.GREEN + f"Columns converted in Online Retail Data: {', '.join(columns_converted)}" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "No data type conversions were necessary for Online Retail Data." + Style.RESET_ALL)

    return df

def convert_campaign_data_types(df):
    print(Fore.CYAN + "\nStarting data type conversions for Campaign Response Data..." + Style.RESET_ALL)
    columns_converted = []

    # Convert CustomerID to string if it's numeric
    try:
        old_dtype = df['CustomerID'].dtype
        if old_dtype == 'int64' or old_dtype == 'float64':
            df['CustomerID'] = df['CustomerID'].astype(str)
            new_dtype = df['CustomerID'].dtype
            describe_conversion('CustomerID', old_dtype, new_dtype)
            columns_converted.append('CustomerID')
    except Exception as e:
        print(Fore.RED + f"Error converting 'CustomerID': {e}" + Style.RESET_ALL)

    # Convert response and loyalty to boolean
    try:
        old_dtype = df['response'].dtype
        if old_dtype != 'bool':
            df['response'] = df['response'].astype(bool)
            new_dtype = df['response'].dtype
            describe_conversion('response', old_dtype, new_dtype)
            columns_converted.append('response')
    except Exception as e:
        print(Fore.RED + f"Error converting 'response': {e}" + Style.RESET_ALL)

    try:
        old_dtype = df['loyalty'].dtype
        if old_dtype != 'bool':
            df['loyalty'] = df['loyalty'].astype(bool)
            new_dtype = df['loyalty'].dtype
            describe_conversion('loyalty', old_dtype, new_dtype)
            columns_converted.append('loyalty')
    except Exception as e:
        print(Fore.RED + f"Error converting 'loyalty': {e}" + Style.RESET_ALL)

    # Handle missing values in 'nps' and convert to ordinal categorical
    try:
        df['nps'] = df['nps'].replace(' ', pd.NA)  # Replace spaces with NA
        df['nps'] = pd.to_numeric(df['nps'], errors='coerce')

        old_dtype = df['nps'].dtype
        df['nps'] = pd.Categorical(df['nps'], categories=range(0, 11), ordered=True)
        new_dtype = df['nps'].dtype
        describe_conversion('nps', old_dtype, new_dtype)
        columns_converted.append('nps')
    except Exception as e:
        print(Fore.RED + f"Error converting 'nps': {e}" + Style.RESET_ALL)

    # Report missing values in nps (only NaN values)
    try:
        missing_nps = df['nps'].isna().sum()
        if missing_nps > 0:
            print(Fore.RED + f"Warning: 'nps' column has {format_number(missing_nps)} missing values (NaN or space characters)." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error checking for missing values in 'nps': {e}" + Style.RESET_ALL)

    if columns_converted:
        print(Fore.GREEN + f"Columns converted in Campaign Response Data: {', '.join(columns_converted)}" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "No data type conversions were necessary for Campaign Response Data." + Style.RESET_ALL)

    return df

def print_data_overview(df, dataset_name):
    try:
        print(Fore.CYAN + f"\nData Overview for {dataset_name}:" + Style.RESET_ALL)
        print(df.dtypes)
        print("\nFirst few rows:")
        print(df.head().to_string(index=False))
    except Exception as e:
        print(Fore.RED + f"Error printing data overview for {dataset_name}: {e}" + Style.RESET_ALL)