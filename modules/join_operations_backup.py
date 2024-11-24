import pandas as pd
import os
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
    joined_data = pd.merge(customer_features, campaign_response_data, on=join_key, how='left')
    
    # Step 2: Display the join result to the user
    format_message(f"Customer features successfully joined with campaign response data using '{join_key}' as the join key.", Fore.CYAN)
    format_message(f"\nJoin Type: Left Join", Fore.GREEN)
    
    # Step 3: List the columns and their data types after the join
    console.print(f"\n[cyan]Joined Dataset: [green]{joined_data.shape[0]} rows, {joined_data.shape[1]} columns.[/green]")
    
    table = Table(title="Joined Dataset Columns and Data Types", show_lines=True)
    table.add_column("Column", justify="left", style="cyan", no_wrap=True)
    table.add_column("Data Type", justify="left", style="green")
    
    for col, dtype in zip(joined_data.columns, joined_data.dtypes):
        table.add_row(col, str(dtype))
    
    console.print(table)

    return joined_data

def derive_customer_features(joined_data):
    """Derive customer-level features and return them in a separate dataset."""
    # Ensure 'TotalValue' is calculated correctly
    if 'TotalValue' not in joined_data.columns:
        joined_data['TotalValue'] = joined_data['Quantity'] * joined_data['UnitPrice']
    
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

    # Step 2: Calculate additional features
    customer_features['avg_order_value'] = customer_features['total_spending'] / customer_features['num_orders']
    customer_features['avg_quantity_per_order'] = customer_features['total_quantity'] / customer_features['num_orders']
    customer_features['customer_tenure'] = (customer_features['last_purchase'] - customer_features['first_purchase']).dt.days
    customer_features['recency'] = (joined_data['InvoiceDate'].max() - customer_features['last_purchase']).dt.days
    customer_features['order_frequency'] = customer_features['customer_tenure'] / customer_features['num_orders']
    customer_features['product_diversity'] = customer_features['num_products_purchased']
    
    # Return behavior: Ratio of negative quantities to total quantities
    customer_returns = joined_data[joined_data['Quantity'] < 0].groupby('CustomerID')['Quantity'].sum().abs()
    customer_features['return_rate'] = customer_features['total_quantity'].map(customer_returns).fillna(0)

    # Step 4: Show a sample of the new dataset with customer-level features
    console.print("\n[cyan]Customer-Level Features (First 5 Rows):[/cyan]")
    customer_table = Table(title="Customer-Level Features Overview", show_lines=True)
    customer_table.add_column("CustomerID", justify="center", style="cyan", no_wrap=True)
    customer_table.add_column("First Purchase", justify="right", style="green")
    customer_table.add_column("Last Purchase", justify="right", style="green")
    customer_table.add_column("Total Spending", justify="right", style="green")
    customer_table.add_column("Number of Orders", justify="right", style="green")
    customer_table.add_column("Average Order Value", justify="right", style="green")
    customer_table.add_column("Recency", justify="right", style="green")
    customer_table.add_column("Customer Tenure", justify="right", style="green")
    customer_table.add_column("Average Quantity per Order", justify="right", style="green")
    customer_table.add_column("Order Frequency", justify="right", style="green")
    customer_table.add_column("Product Diversity", justify="right", style="green")
    customer_table.add_column("Return Rate", justify="right", style="green")
    
    # Populate the table with first 5 rows
    for _, row in customer_features.head().iterrows():
        customer_table.add_row(str(row['CustomerID']), 
                               str(row['first_purchase'].date()),
                               str(row['last_purchase'].date()),
                               f"{row['total_spending']:.2f}", 
                               str(row['num_orders']), 
                               f"{row['avg_order_value']:.2f}", 
                               str(row['recency']),
                               str(row['customer_tenure']),
                               f"{row['avg_quantity_per_order']:.2f}",
                               f"{row['order_frequency']:.2f}",
                               str(row['product_diversity']),
                               f"{row['return_rate']:.2f}")
    console.print(customer_table)

    return customer_features

if __name__ == "__main__":
    # Example datasets to demonstrate functionality
    online_retail_data = pd.read_csv(os.path.join('path_to_your_data', 'online_retail.csv'))
    campaign_response_data = pd.read_csv(os.path.join('path_to_your_data', 'campaign_response.csv'))

    customer_features = derive_customer_features(online_retail_data)
    joined_data = perform_customer_campaign_join(customer_features, campaign_response_data)