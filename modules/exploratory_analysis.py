import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from rich.console import Console
from rich.table import Table
from colorama import Fore, Style

def analyze_data(data, response='response'):
    """
    Provide options for different EDA visualizations, including heatmaps, QQ plots, boxplots, response rate tables,
    and bar charts/histograms.
    """
    while True:
        print(Fore.CYAN + "\nExploratory Data Analysis Menu:")
        print(Fore.YELLOW + "1. Generate Correlation Heatmap")
        print(Fore.YELLOW + "2. Generate QQ Plots (for all numerical variables)")
        print(Fore.YELLOW + "3. Generate Boxplots")
        print(Fore.YELLOW + "4. Generate Response Rate Table")
        print(Fore.YELLOW + "5. Generate Bar Charts and Histograms")
        print(Fore.YELLOW + "6. Analyze Month-on-Month Sales")
        print(Fore.YELLOW + "7. Identify Top 5 Products Based on Sales")
        print(Fore.YELLOW + "0. Return to main menu" + Style.RESET_ALL)

        choice = input(Fore.CYAN + "Choose an option: " + Style.RESET_ALL)

        if choice == '1':
            generate_correlation_heatmap(data, response)

        elif choice == '2':
            # Prompt user about outliers for QQ plots
            outlier_choice = input(Fore.CYAN + "Include outliers in QQ plots? (yes/no): " + Style.RESET_ALL).strip().lower()
            include_outliers = outlier_choice == 'yes'
            generate_qq_plots_matrix(data, outliers_included=include_outliers)

        elif choice == '3':
            generate_boxplots_by_response(data)

        elif choice == '4':
            generate_response_rate_table(data, response)

        elif choice == '5':
            generate_bar_charts_and_histograms(data, response)

        elif choice == '6':  # Call the new function
            analyze_monthly_sales_from_original()

        if choice == '7':
            identify_top_products()

        elif choice == '0':
            print(Fore.CYAN + "Returning to main menu..." + Style.RESET_ALL)
            break

        else:
            print(Fore.RED + "Invalid choice, please select a valid option." + Style.RESET_ALL)

def generate_correlation_heatmap(data, response):
    """Generate a heatmap for all numerical variables and the boolean response."""
    print(Fore.CYAN + "\nGenerating Correlation Heatmap with Response..." + Style.RESET_ALL)
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title(f'Correlation Heatmap (Including {response})')
    plt.show()

def generate_qq_plots_matrix(data, outliers_included):
    """
    Generate a matrix of QQ plots for all numerical variables to assess normality, 
    without axis labels (only showing the variable names).
    :param data: DataFrame containing the data for analysis
    :param outliers_included: Boolean indicating whether outliers should be kept or removed
    """
    # Define the numerical variables to plot
    numerical_vars = data.select_dtypes(include=['float64', 'int64']).columns.tolist()  # Focus only on numerical variables

    print(Fore.CYAN + "Debug: Checking available columns for QQ plots." + Style.RESET_ALL)
    available_columns = data.columns
    print(Fore.CYAN + f"Available columns: {list(available_columns)}" + Style.RESET_ALL)

    # Remove outliers if requested by the user
    if not outliers_included:
        data = exclude_outliers_iqr(data, numerical_vars)

    # Set up the grid for QQ plots (3 columns for better visualization)
    num_vars = len(numerical_vars)
    rows = (num_vars + 2) // 3  # Calculate the number of rows needed for the grid
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    fig.suptitle('QQ Plots (Normality Check)', fontsize=16)

    # Flatten axes array for easier iteration
    axes = axes.flatten()

    # Generate QQ plots for each numerical variable
    for i, var in enumerate(numerical_vars):
        if var in available_columns:
            stats.probplot(data[var].dropna(), dist="norm", plot=axes[i])
            axes[i].set_title(f'QQ Plot for {var}', fontsize=12)
            axes[i].set_xlabel('')  # Remove x-axis label
            axes[i].set_ylabel('')  # Remove y-axis label
        else:
            print(Fore.RED + f"Warning: Column '{var}' not found in the dataset." + Style.RESET_ALL)
            axes[i].set_visible(False)  # Hide any unused subplots

    # Adjust layout for better visualization
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def generate_bar_plot(data, variable, response):
    """
    Generates a bar plot for categorical variables, showing the breakdown by the response variable.
    """
    print(Fore.CYAN + f"\nGenerating Bar Plot for {variable}, grouped by {response}..." + Style.RESET_ALL)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=variable, hue=response, data=data)
    plt.title(f'{variable} by {response}')
    plt.show()

def generate_boxplots_by_response(data, response='response'):
    """
    Allows the user to select which variable they want to generate a boxplot for, with the option
    to include or exclude outliers, with a breakdown by the response variable.
    """
    numerical_vars = [
        'total_spending', 'num_orders', 'avg_order_value', 'total_quantity',
        'customer_tenure', 'order_frequency', 'product_diversity', 'return_rate',
        'avg_quantity_per_order', 'recency', 'n_comp', 'n_communications'
    ]

    categorical_vars = ['loyalty', 'nps']

    while True:
        print(Fore.CYAN + "\nBoxplot Menu (Breakdown by Response):")
        for idx, var in enumerate(numerical_vars + categorical_vars, start=1):
            print(Fore.YELLOW + f"{idx}. Plot for {var}")
        print(Fore.YELLOW + "0. Return to the previous menu" + Style.RESET_ALL)

        choice = input(Fore.CYAN + "Choose a variable to visualize with a boxplot (0 to return): " + Style.RESET_ALL).strip()

        if choice.isdigit() and int(choice) in range(len(numerical_vars + categorical_vars) + 1):
            if int(choice) == 0:
                print(Fore.CYAN + "Returning to the previous menu..." + Style.RESET_ALL)
                break

            selected_var = (numerical_vars + categorical_vars)[int(choice) - 1]
            # Handle numerical variables with boxplots
            if selected_var in numerical_vars:
                outlier_choice = input(Fore.CYAN + f"Include outliers in the boxplot for {selected_var}? (yes/no): " + Style.RESET_ALL).strip().lower()

                # Remove outliers if the user chooses not to include them
                if outlier_choice == 'no':
                    data_clean = exclude_outliers_iqr(data, [selected_var])
                else:
                    data_clean = data

                # Generate the boxplot for numerical variables
                generate_boxplot(data_clean, selected_var, response)

            # Handle categorical variables with bar plots
            elif selected_var in categorical_vars:
                generate_bar_plot(data, selected_var, response)

        else:
            print(Fore.RED + "Invalid choice, please select a valid option." + Style.RESET_ALL)

def generate_boxplot(data, variable, response):
    """
    Generates a boxplot for a selected variable with a breakdown by the response variable.
    """
    print(Fore.CYAN + f"\nGenerating Boxplot for {variable}, grouped by {response}..." + Style.RESET_ALL)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=response, y=variable, data=data)
    plt.title(f'{variable} by {response}')
    plt.show()

def generate_stacked_histograms(data, variables, response):
    """Generate stacked histograms for all numerical variables split by response."""
    print(Fore.CYAN + "\nGenerating Stacked Histograms..." + Style.RESET_ALL)
    for var in variables:
        sns.histplot(data, x=var, hue=response, multiple="stack", bins=30)
        plt.title(f'{var} Distribution Split by {response}')
        plt.show()

def exclude_outliers_iqr(df, columns):
    """Temporarily exclude outliers based on IQR for the selected columns."""
    df_clean = df.copy()
    for col in columns:
        if col not in df_clean.columns:
            print(Fore.RED + f"Column '{col}' does not exist in the dataset. Skipping outlier removal for this column." + Style.RESET_ALL)
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def generate_response_rate_table(data, response):
    """
    Generate and display a table of response rates for loyalty and NPS categories using Rich.
    """
    print(Fore.CYAN + "\nGenerating Response Rate Table..." + Style.RESET_ALL)

    # Initialize the console object
    console = Console()

    # Table for Loyalty
    if 'loyalty' in data.columns:
        loyalty_table = data.groupby('loyalty').agg(
            N=('response', 'size'),
            N_plus=('response', 'sum')
        ).reset_index()
        loyalty_table['Response Rate (%)'] = (loyalty_table['N_plus'] / loyalty_table['N']) * 100

        # Display with Rich
        rich_table = Table(title="Response Rate by Loyalty", show_lines=True)
        rich_table.add_column("Loyalty", justify="center", style="cyan")
        rich_table.add_column("N (Total)", justify="center", style="magenta")
        rich_table.add_column("N+ (Responded)", justify="center", style="green")
        rich_table.add_column("Response Rate (%)", justify="center", style="yellow")
        for _, row in loyalty_table.iterrows():
            rich_table.add_row(str(row['loyalty']), str(row['N']), str(row['N_plus']), f"{row['Response Rate (%)']:.2f}")
        console.print(rich_table)

    # Table for NPS Categories
    if 'nps' in data.columns:
        # Define NPS categories
        bins = [0, 6, 8, 10]
        labels = ['Detractor (0-6)', 'Passive (7-8)', 'Promoter (9-10)']
        data['NPS Category'] = pd.cut(data['nps'].cat.codes, bins=bins, labels=labels, right=True)

        nps_table = data.groupby('NPS Category').agg(
            N=('response', 'size'),
            N_plus=('response', 'sum')
        ).reset_index()
        nps_table['Response Rate (%)'] = (nps_table['N_plus'] / nps_table['N']) * 100

        # Display with Rich
        rich_table = Table(title="Response Rate by NPS Category", show_lines=True)
        rich_table.add_column("NPS Category", justify="center", style="cyan")
        rich_table.add_column("N (Total)", justify="center", style="magenta")
        rich_table.add_column("N+ (Responded)", justify="center", style="green")
        rich_table.add_column("Response Rate (%)", justify="center", style="yellow")
        for _, row in nps_table.iterrows():
            rich_table.add_row(str(row['NPS Category']), str(row['N']), str(row['N_plus']), f"{row['Response Rate (%)']:.2f}")
        console.print(rich_table)

def generate_bar_charts_and_histograms(data, response):
    """
    Generate bar charts and histograms for selected customer features, including additional derived variables.
    """
    print(Fore.CYAN + "\nBar Charts and Histograms Menu:")
    print(Fore.YELLOW + "1. Total Sales by Response")
    print(Fore.YELLOW + "2. Unique Products Purchased by Response")
    print(Fore.YELLOW + "3. Number of Invoices by Response")
    print(Fore.YELLOW + "4. Average Order Value by Response")
    print(Fore.YELLOW + "5. Customer Tenure by Response")
    print(Fore.YELLOW + "6. Order Frequency by Response")  # Updated from Return Rate
    print(Fore.YELLOW + "0. Return to the previous menu" + Style.RESET_ALL)

    while True:
        choice = input(Fore.CYAN + "Choose an option: " + Style.RESET_ALL).strip()

        if choice == '1':
            generate_bar_chart(data, feature='total_spending', response=response)

        elif choice == '2':
            generate_bar_chart(data, feature='num_products_purchased', response=response)

        elif choice == '3':
            generate_bar_chart(data, feature='num_orders', response=response)

        elif choice == '4':
            generate_bar_chart(data, feature='avg_order_value', response=response)

        elif choice == '5':
            generate_bar_chart(data, feature='customer_tenure', response=response)

        elif choice == '6':
            generate_bar_chart(data, feature='order_frequency', response=response)

        elif choice == '0':
            print(Fore.CYAN + "Returning to the previous menu..." + Style.RESET_ALL)
            break

        else:
            print(Fore.RED + "Invalid choice, please select a valid option." + Style.RESET_ALL)

def generate_bar_chart(data, feature, response):
    """
    Generate a bar chart for a selected feature, grouped by the response variable.
    """
    print(Fore.CYAN + f"\nGenerating Bar Chart for {feature} by {response}..." + Style.RESET_ALL)

    # Aggregate data by response
    aggregated_data = data.groupby(response)[feature].agg(['mean', 'median']).reset_index()

    # Plot mean and median as a bar chart
    aggregated_data.plot(
        kind='bar',
        x=response,
        y=['mean', 'median'],
        figsize=(8, 6),
        title=f'{feature.capitalize()} by {response}',
    )
    plt.ylabel(feature.capitalize())
    plt.grid(alpha=0.3)
    plt.show()

def analyze_monthly_sales_from_original():
    """
    Analyze and visualize month-on-month sales from the original dataset.
    Includes data integrity checks, aggregation, and filtering by a cutoff date.
    """
    print(Fore.CYAN + "\nAnalyzing Month-on-Month Sales from Original Dataset..." + Style.RESET_ALL)
    file_path = "./datasets/online_retail.csv"

    # Step 1: Load the dataset
    try:
        data = pd.read_csv(file_path)
        print(Fore.CYAN + "Original dataset loaded successfully." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error: Could not load dataset from {file_path}. Details: {e}" + Style.RESET_ALL)
        return

    # Step 2: Check for required columns
    required_columns = ['InvoiceDate', 'Quantity', 'UnitPrice']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(Fore.RED + f"Error: Missing required columns: {missing_columns}. Please check the dataset." + Style.RESET_ALL)
        print(Fore.YELLOW + f"Available columns: {list(data.columns)}" + Style.RESET_ALL)
        return

    # Step 3: Handle duplicates and missing values
    initial_rows = data.shape[0]
    data.drop_duplicates(inplace=True)
    print(Fore.YELLOW + f"Removed {initial_rows - data.shape[0]} duplicate rows." + Style.RESET_ALL)

    data.dropna(subset=required_columns, inplace=True)
    print(Fore.YELLOW + f"Removed rows with missing values in {required_columns}." + Style.RESET_ALL)

    # Step 4: Parse InvoiceDate and aggregate sales
    try:
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    except Exception as e:
        print(Fore.RED + f"Error: Failed to convert 'InvoiceDate' to datetime. Details: {e}" + Style.RESET_ALL)
        return

    # Ensure Quantity and UnitPrice are numeric
    try:
        data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
        data['UnitPrice'] = pd.to_numeric(data['UnitPrice'], errors='coerce')
    except Exception as e:
        print(Fore.RED + f"Error: Failed to ensure numeric values for Quantity or UnitPrice. Details: {e}" + Style.RESET_ALL)
        return

    # Calculate sales and derive month
    data['Sales'] = data['Quantity'] * data['UnitPrice']
    data['Month'] = data['InvoiceDate'].dt.to_period('M')

    # Step 5: Aggregate by month
    monthly_sales = data.groupby('Month')['Sales'].sum().reset_index()
    monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()  # Convert to timestamp for plotting

    # Step 6: Filter data to exclude incomplete months
    cutoff_date = "2023-11-30"  # Set the cutoff date to avoid representing interrupted data collection
    monthly_sales = monthly_sales[monthly_sales['Month'] < pd.Timestamp(cutoff_date)]

    # Debugging output: Display aggregated sales
    print(Fore.CYAN + "\nDebug: Aggregated Monthly Sales Data:")
    print(monthly_sales)

    # Inform the user about the steps taken
    print(Fore.CYAN + "\nSteps Performed:" + Style.RESET_ALL)
    print(Fore.YELLOW + "- Duplicates removed." + Style.RESET_ALL)
    print(Fore.YELLOW + "- Missing values handled for InvoiceDate, Quantity, and UnitPrice." + Style.RESET_ALL)
    print(Fore.YELLOW + "- Aggregated Sales = Quantity * UnitPrice, including 0 and negative values." + Style.RESET_ALL)
    print(Fore.YELLOW + f"- Excluded data after {cutoff_date}." + Style.RESET_ALL)

    # Step 7: Plot the monthly sales trend
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o')
    plt.title('Month-on-Month Sales Trend (up to Nov 2023)')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.grid(alpha=0.3)
    plt.show()

def identify_top_products():
    """
    Identify the top 5 products based on sales and visualize the results.
    """
    console = Console()
    try:
        # Load the dataset
        file_path = "./datasets/online_retail.csv"
        data = pd.read_csv(file_path)
        console.print(f"Dataset loaded successfully from {file_path}.", style="bold green")

        # Preliminary data integrity checks
        console.print("Performing preliminary data checks...", style="bold cyan")
        initial_rows = data.shape[0]

        # Drop duplicates
        data.drop_duplicates(inplace=True)
        console.print(f"Removed duplicates: {initial_rows - data.shape[0]} rows dropped.", style="bold cyan")

        # Drop rows with missing critical fields
        required_columns = ['StockCode', 'Description', 'Quantity', 'UnitPrice']
        missing_before = data.isna().sum().sum()
        data.dropna(subset=required_columns, inplace=True)
        missing_after = data.isna().sum().sum()
        console.print(f"Removed rows with missing values in {required_columns}: {missing_before - missing_after} rows dropped.", style="bold cyan")

        # Include zero and negative values
        data['TotalSales'] = data['Quantity'] * data['UnitPrice']

        # Group by product and calculate total sales
        aggregated_data = data.groupby(['StockCode', 'Description'], as_index=False)['TotalSales'].sum()

        # Sort and select the top 5 products
        top_products = aggregated_data.nlargest(5, 'TotalSales')

        # Inform the user about top products
        console.print("\nTop 5 Products by Sales:", style="bold green")
        table = Table(title="Top 5 Products")
        table.add_column("StockCode", justify="center", style="cyan", no_wrap=True)
        table.add_column("Description", justify="center", style="magenta")
        table.add_column("Total Sales", justify="right", style="green")
        for _, row in top_products.iterrows():
            table.add_row(str(row['StockCode']), row['Description'], f"${row['TotalSales']:.2f}")
        console.print(table)

        # Plot the top 5 products
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_products,
            x='TotalSales',
            y='Description',
            palette="Blues_d"
        )
        plt.title("Top 5 Products by Sales")
        plt.xlabel("Total Sales")
        plt.ylabel("Product Description")
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        console.print(f"Error during analysis: {str(e)}", style="bold red")