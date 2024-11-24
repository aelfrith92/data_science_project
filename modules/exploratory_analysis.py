import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from colorama import Fore, Style

def analyze_data(data, response='response'):
    """
    Provide options for different EDA visualizations, including heatmaps and QQ plot matrix.
    """
    while True:
        print(Fore.CYAN + "\nExploratory Data Analysis Menu:")
        print(Fore.YELLOW + "1. Generate Correlation Heatmap")
        print(Fore.YELLOW + "2. Generate QQ Plots (for all numerical variables)")
        print(Fore.YELLOW + "3. Generate Boxplots")
        print(Fore.YELLOW + "0. Return to main menu" + Style.RESET_ALL)

        choice = input(Fore.CYAN + "Choose an option: " + Style.RESET_ALL)

        if choice == '1':
            generate_correlation_heatmap(data, response)
        
        elif choice == '2':
            outlier_choice = input(Fore.CYAN + "Include outliers in QQ plots? (yes/no): " + Style.RESET_ALL).strip().lower()
            if outlier_choice == 'no':
                generate_qq_plots_matrix(data, outliers_included=False)
            else:
                generate_qq_plots_matrix(data, outliers_included=True)

        elif choice == '3':
            generate_boxplots_by_response(data)

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
        'avg_quantity_per_order', 'recency', 'n_comp'
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