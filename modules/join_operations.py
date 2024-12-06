import pandas as pd
import os
import matplotlib.pyplot as plt
from colorama import Fore, Style
from rich.console import Console
from rich.table import Table

console = Console()

def format_message(message, color=Fore.GREEN):
    """Utility function to format and print messages with color."""
    print(color + message + Style.RESET_ALL)

def perform_customer_campaign_join(customer_features, campaign_response_data):
    """Perform the join between customer features and campaign response data."""
    # Step 1: Join customer_features with campaign_response_data on CustomerID
    join_key = 'CustomerID'
    joined_data = pd.merge(customer_features, campaign_response_data, on=join_key, how='inner')

    # Step 2: Display the join result to the user
    format_message(f"Customer features successfully joined with campaign response data using '{join_key}' as the join key.", Fore.CYAN)
    format_message(f"\nJoin Type: Inner Join", Fore.GREEN)

    # Step 3: Calculate and display campaign response rate
    total_responses = campaign_response_data['response'].sum()
    total_records = len(campaign_response_data)
    response_rate = (total_responses / total_records) * 100
    non_response_rate = 100 - response_rate

    print(Fore.CYAN + f"\nOverall Campaign Response Rate: {response_rate:.2f}%" + Style.RESET_ALL)
    print(Fore.CYAN + f"Overall Non-Response Rate: {non_response_rate:.2f}%" + Style.RESET_ALL)

    # Step 4: Plot a pie chart for response rates
    plt.figure(figsize=(6, 6))
    labels = ['Responded', 'Did Not Respond']
    sizes = [total_responses, total_records - total_responses]
    colors = ['#4CAF50', '#FF5252']
    explode = (0.1, 0)  # Highlight the 'Responded' slice
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
    plt.title("Campaign Response Distribution")
    plt.show()

    # Step 5: List the columns and their data types after the join
    console.print(f"\n[cyan]Joined Dataset: [green]{joined_data.shape[0]} rows, {joined_data.shape[1]} columns.[/green]")

    table = Table(title="Joined Dataset Columns and Data Types", show_lines=True)
    table.add_column("Column", justify="left", style="cyan", no_wrap=True)
    table.add_column("Data Type", justify="left", style="green")

    for col, dtype in zip(joined_data.columns, joined_data.dtypes):
        table.add_row(col, str(dtype))

    console.print(table)

    return joined_data

def derive_customer_features(joined_data):
    """Derive customer-level features and return them in a dataset containing both original and binned variables."""
    # Ensure 'TotalValue' is calculated correctly
    if 'TotalValue' not in joined_data.columns:
        joined_data['TotalValue'] = joined_data['Quantity'] * joined_data['UnitPrice']

    # Step 4: Create Binned Categorical Variables with User-Friendly Labels
    def create_categorical_bins(column, column_name):
        """
        Create descriptive bins for a numeric column while excluding outliers.
        Handles NaN, inf, and extreme outliers gracefully.
        """
        if column.isnull().all():
            console.print(f"[red]Column '{column_name}' has all missing values. Assigning 'Unknown' for all rows.[/red]")
            return pd.Series(["Unknown"] * len(column), dtype="category")

        try:
            # Define labels for quartiles
            base_labels = ["Very Low", "Low", "High", "Very High"]

            # Clean the column: replace inf and drop NaN values
            column_clean = column.replace([float('inf'), float('-inf')], pd.NA).dropna()

            if len(column_clean.unique()) < 2:  # Handle case with insufficient unique values
                console.print(f"[yellow]Column '{column_name}' has insufficient unique values for binning. Assigning 'Unknown' for all rows.[/yellow]")
                return pd.Series(["Unknown"] * len(column), dtype="category")

            # Exclude outliers using the IQR method
            Q1 = column_clean.quantile(0.25)
            Q3 = column_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            column_without_outliers = column_clean[(column_clean >= lower_bound) & (column_clean <= upper_bound)]

            if column_without_outliers.empty:
                console.print(f"[yellow]Column '{column_name}' has no data left after outlier removal. Assigning 'Unknown' for all rows.[/yellow]")
                return pd.Series(["Unknown"] * len(column), dtype="category")

            # Generate bins dynamically
            bins = pd.qcut(column_without_outliers, q=4, retbins=True, duplicates="drop")[1]
            labels = base_labels[: len(bins) - 1]  # Adjust labels to match number of bins

            # Create binned column
            binned_column = pd.cut(column, bins=bins, labels=labels, include_lowest=True)

            # Add "Unknown" for missing values
            binned_column = binned_column.cat.add_categories("Unknown").fillna("Unknown")
            return binned_column
        except Exception as e:
            console.print(f"[red]Error binning column '{column_name}': {e}. Assigning 'Unknown' for all rows.[/red]")
            return pd.Series(["Unknown"] * len(column), dtype="category")

    # Step 1: Group by CustomerID to calculate the features
    customer_features = joined_data.groupby('CustomerID').agg({
        'InvoiceDate': ['min', 'max'],              # First and last purchase date
        'InvoiceNo': 'nunique',                     # Number of unique orders
        'Quantity': 'sum',                          # Total quantity
        'TotalValue': 'sum',                        # Total spending (Quantity * UnitPrice)
        'StockCode': 'nunique',                     # Number of products purchased
        'UnitPrice': 'mean'                         # Average price per unit
    }).reset_index()

    # Rename columns
    customer_features.columns = ['CustomerID', 'first_purchase', 'last_purchase', 
                                 'num_orders', 'total_quantity', 'total_spending', 
                                 'num_products_purchased', 'avg_price_per_unit']

    # Step 2: Calculate additional continuous features
    customer_features['avg_order_value'] = customer_features['total_spending'] / customer_features['num_orders']
    customer_features['avg_quantity_per_order'] = customer_features['total_quantity'] / customer_features['num_orders']
    customer_features['customer_tenure'] = (customer_features['last_purchase'] - customer_features['first_purchase']).dt.days
    customer_features['recency'] = (joined_data['InvoiceDate'].max() - customer_features['last_purchase']).dt.days
    customer_features['order_frequency'] = customer_features['customer_tenure'] / customer_features['num_orders']
    customer_features['product_diversity'] = customer_features['num_products_purchased']

    # Include Total Sales Binning
    customer_features['total_sales_bin'] = create_categorical_bins(customer_features['total_spending'], 'total_spending')

    # Apply the updated function to all binned variables
    customer_features['recency_bin'] = create_categorical_bins(customer_features['recency'], 'recency')
    customer_features['total_quantity_bin'] = create_categorical_bins(customer_features['total_quantity'], 'total_quantity')
    customer_features['avg_order_value_bin'] = create_categorical_bins(customer_features['avg_order_value'], 'avg_order_value')
    customer_features['customer_tenure_bin'] = create_categorical_bins(customer_features['customer_tenure'], 'customer_tenure')
    customer_features['order_frequency_bin'] = create_categorical_bins(customer_features['order_frequency'], 'order_frequency')

    # Debugging and Validation
    for binned_column in [
        'recency_bin', 'total_quantity_bin', 'avg_order_value_bin',
        'customer_tenure_bin', 'order_frequency_bin', 'total_sales_bin'
    ]:
        print(f"Updated {binned_column} values after re-binning:")
        print(customer_features[binned_column].value_counts(dropna=False))
        print(f"Missing value counts for {binned_column}: {customer_features[binned_column].isna().sum()}")

    # Step 3: Calculate return_rate as the ratio of negative quantities to total quantities
    customer_returns = joined_data[joined_data['Quantity'] < 0].groupby('CustomerID')['Quantity'].sum().abs()
    customer_features['return_rate'] = customer_features['CustomerID'].map(customer_returns) / customer_features['total_quantity']
    customer_features['return_rate'] = customer_features['return_rate'].fillna(0)  # Handle missing values

    # # Step 4: Create Binned Categorical Variables (preserving the originals)
    # def create_bins(column, column_name):
    #     no_outliers = column[column < column.quantile(0.95)]  # Exclude outliers
    #     if len(no_outliers) < 2:  # Ensure sufficient data for binning
    #         console.print(f"[red]Insufficient data for binning {column_name}. Skipping binning.[/red]")
    #         return pd.NA
    #     bins = pd.qcut(no_outliers, q=4, duplicates='drop', retbins=True)[1]
    #     return pd.cut(column, bins=bins, include_lowest=True)

    # customer_features['recency_bin'] = create_bins(customer_features['recency'], 'recency')
    # customer_features['total_quantity_bin'] = create_bins(customer_features['total_quantity'], 'total_quantity')
    # customer_features['avg_order_value_bin'] = create_bins(customer_features['avg_order_value'], 'avg_order_value')
    # customer_features['customer_tenure_bin'] = create_bins(customer_features['customer_tenure'], 'customer_tenure')
    # customer_features['order_frequency_bin'] = create_bins(customer_features['order_frequency'], 'order_frequency')

    # Apply the updated function to all binned variables
    customer_features['recency_bin'] = create_categorical_bins(customer_features['recency'], 'recency')
    customer_features['total_quantity_bin'] = create_categorical_bins(customer_features['total_quantity'], 'total_quantity')
    customer_features['avg_order_value_bin'] = create_categorical_bins(customer_features['avg_order_value'], 'avg_order_value')
    customer_features['customer_tenure_bin'] = create_categorical_bins(customer_features['customer_tenure'], 'customer_tenure')
    customer_features['order_frequency_bin'] = create_categorical_bins(customer_features['order_frequency'], 'order_frequency')

    # Debugging and Validation
    for binned_column in ['recency_bin', 'total_quantity_bin', 'avg_order_value_bin', 
                        'customer_tenure_bin', 'order_frequency_bin']:
        print(f"Updated {binned_column} values after re-binning:")
        print(customer_features[binned_column].value_counts(dropna=False))
        print(f"Missing value counts for {binned_column}: {customer_features[binned_column].isna().sum()}")

    # Step 5: Ensure the final dataset contains both original and binned variables
    customer_features = customer_features[[
        'CustomerID', 'first_purchase', 'last_purchase', 'total_spending', 'num_orders',
        'total_quantity', 'avg_order_value', 'recency', 'customer_tenure',
        'avg_quantity_per_order', 'order_frequency', 'product_diversity', 'return_rate',
        # Include binned variables
        'recency_bin', 'total_quantity_bin', 'avg_order_value_bin',
        'customer_tenure_bin', 'order_frequency_bin', 'total_sales_bin'
    ]]

    # Step 6: Display updated dataset with Total Sales Bin
    customer_table = Table(title="Customer-Level Features Overview", show_lines=True)

    # Add columns explicitly
    customer_table.add_column("CustomerID", justify="center", style="cyan", no_wrap=True)
    customer_table.add_column("First Purchase", justify="right", style="green")
    customer_table.add_column("Last Purchase", justify="right", style="green")
    customer_table.add_column("Total Spending", justify="right", style="green")
    customer_table.add_column("Number of Orders", justify="right", style="green")
    customer_table.add_column("Average Order Value", justify="right", style="green")
    customer_table.add_column("Recency", justify="right", style="green")
    customer_table.add_column("Recency Bin", justify="right", style="magenta")
    customer_table.add_column("Customer Tenure", justify="right", style="green")
    customer_table.add_column("Customer Tenure Bin", justify="right", style="magenta")
    customer_table.add_column("Total Quantity", justify="right", style="green")
    customer_table.add_column("Total Quantity Bin", justify="right", style="magenta")
    customer_table.add_column("Order Frequency", justify="right", style="green")
    customer_table.add_column("Order Frequency Bin", justify="right", style="magenta")
    customer_table.add_column("Return Rate", justify="right", style="green")
    customer_table.add_column("Total Sales Bin", justify="right", style="magenta")

    # Populate the table with rows
    for _, row in customer_features.head().iterrows():
        customer_table.add_row(
            str(row['CustomerID']),
            str(row['first_purchase'].date()),
            str(row['last_purchase'].date()),
            f"{row['total_spending']:.2f}",
            str(row['num_orders']),
            f"{row['avg_order_value']:.2f}",
            str(row['recency']),
            str(row['recency_bin']) if not pd.isna(row['recency_bin']) else "N/A",
            str(row['customer_tenure']),
            str(row['customer_tenure_bin']) if not pd.isna(row['customer_tenure_bin']) else "N/A",
            str(row['total_quantity']),
            str(row['total_quantity_bin']) if not pd.isna(row['total_quantity_bin']) else "N/A",
            f"{row['order_frequency']:.2f}",
            str(row['order_frequency_bin']) if not pd.isna(row['order_frequency_bin']) else "N/A",
            f"{row['return_rate']:.2f}",
            str(row['total_sales_bin']) if not pd.isna(row['total_sales_bin']) else "N/A"
        )

    # Print the table
    console.print(customer_table)

    return customer_features

if __name__ == "__main__":
    # Example datasets to demonstrate functionality
    online_retail_data = pd.read_csv(os.path.join('path_to_your_data', 'online_retail.csv'))
    campaign_response_data = pd.read_csv(os.path.join('path_to_your_data', 'campaign_response.csv'))

    customer_features = derive_customer_features(online_retail_data)
    joined_data = perform_customer_campaign_join(customer_features, campaign_response_data)