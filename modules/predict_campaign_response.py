from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style, init


def preprocess_data_for_model(data):
    """
    Preprocess the dataset by converting datetime columns, handling categorical variables,
    replacing missing values, and providing a detailed log of operations.
    """
    data = data.copy()
    log_messages = []

    # Handle datetime columns
    if 'first_purchase' in data.columns and 'last_purchase' in data.columns:
        data['customer_tenure_days'] = (data['last_purchase'] - data['first_purchase']).dt.days
        data['recency_days'] = (data['last_purchase'].max() - data['last_purchase']).dt.days
        data.drop(columns=['first_purchase', 'last_purchase'], inplace=True)
        log_messages.append("Converted 'first_purchase' and 'last_purchase' to 'customer_tenure_days' and 'recency_days'.")

    # Convert categorical variables
    if 'loyalty' in data.columns:
        data['loyalty'] = data['loyalty'].astype(int)
        log_messages.append("Converted 'loyalty' to an integer type.")
    if 'nps' in data.columns:
        data['nps'] = data['nps'].astype('category').cat.codes
        log_messages.append("Converted 'nps' to ordinal numeric codes.")

    # Replace infinities with NaN
    inf_count = data.isin([float('inf'), float('-inf')]).sum().sum()
    data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    log_messages.append(f"Replaced {inf_count} infinite values with NaN.")
    
    pd.set_option('future.no_silent_downcasting', True)
    
    # Fill NaN values explicitly
    nan_count_before = data.isna().sum().sum()
    data = data.fillna(0)
    nan_count_after = data.isna().sum().sum()
    log_messages.append(f"Filled {nan_count_before - nan_count_after} NaN values with 0.")

    # Explicitly handle object columns
    def safe_to_numeric(series):
        """Convert a pandas Series to numeric if possible, otherwise return as is."""
        try:
            return pd.to_numeric(series)
        except ValueError:
            return series

    object_columns = data.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        data = data.apply(
            lambda col: safe_to_numeric(col) if col.dtypes == 'object' else col
        )
        log_messages.append(f"Processed and converted {len(object_columns)} object columns to numeric types if possible: {object_columns}")

    # Print detailed log
    print(Fore.CYAN + "\nPreprocessing Log:" + Style.RESET_ALL)
    for message in log_messages:
        print(Fore.GREEN + "✔ " + message + Style.RESET_ALL)

    # Return the preprocessed data
    return data

def predict_customer_response(data, response='response'):
    """
    Predict customer response using Logistic Regression and Random Forest models.
    Provides options for preprocessing, feature significance exploration, multicollinearity checks, and model insights.
    """
    print(Fore.CYAN + "\nPredicting Customer Response..." + Style.RESET_ALL)

    def submenu():
        """Submenu for configuring model parameters and running actions."""
        while True:
            print(Fore.CYAN + "\nCustomer Response Prediction Submenu:" + Style.RESET_ALL)
            print("1. Preprocess Data")
            print("2. Explore Feature Significance")
            print("3. Check Multicollinearity")
            print("4. Run Models")
            print("0. Return to Main Menu")

            choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)

            if choice == '1':
                processed_data = preprocess_data_for_model(data)
                print(Fore.GREEN + "Data preprocessing completed. Ready for modeling." + Style.RESET_ALL)
            elif choice == '2':
                visualize_significant_variables(data, response)
            elif choice == '3':
                check_and_handle_multicollinearity(data, response)
            elif choice == '4':
                run_models(data, response)
            elif choice == '0':
                print(Fore.CYAN + "Returning to Main Menu..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Invalid choice! Please try again." + Style.RESET_ALL)

    def run_models(data, response):
        """
        Train and evaluate Logistic Regression and Random Forest models.
        Log and visualize variable details and data types used in the model.
        """
        processed_data = preprocess_data_for_model(data)
        X = processed_data.drop(columns=[response, 'CustomerID'])
        y = processed_data[response]

        # Log variables
        variables_log = {"Kept": [], "Excluded": []}

        # Prompt user for train/test split
        while True:
            try:
                train_size = float(input(Fore.YELLOW + "Enter the training percentage (e.g., 0.7 for 70% train): " + Style.RESET_ALL))
                if 0.5 <= train_size <= 0.9:
                    break
                else:
                    print(Fore.RED + "Train/test split must be between 0.5 and 0.9. Please try again." + Style.RESET_ALL)
            except ValueError:
                print(Fore.RED + "Invalid input. Please enter a decimal number (e.g., 0.7)." + Style.RESET_ALL)

        print(Fore.CYAN + "\nModel Run Options:" + Style.RESET_ALL)
        print("1. Use all variables.")
        print("2. Use significant variables only.")
        print("3. Use significant variables (after removing multicollinear ones).")
        choice = input(Fore.YELLOW + "Choose an option: " + Style.RESET_ALL)

        if choice == '2':
            X, excluded_with_reasons = filter_significant_features_with_reasons(X, y)
            variables_log["Kept"] = X.columns.tolist()
            variables_log["Excluded"].extend(excluded_with_reasons)
        elif choice == '3':
            X, excluded_with_reasons = filter_significant_features_with_reasons(X, y)
            variables_log["Kept"] = X.columns.tolist()
            variables_log["Excluded"].extend(excluded_with_reasons)
            X, excluded_multicollinear_with_reasons = remove_multicollinear_features_with_reasons(X)
            variables_log["Excluded"].extend(excluded_multicollinear_with_reasons)

        # Log data types
        data_types = X.dtypes.reset_index()
        data_types.columns = ["Variable", "Data Type"]

        print(Fore.GREEN + "\nVariable Data Types Used in the Model:" + Style.RESET_ALL)
        print(data_types.to_string(index=False))

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        variables_log["Kept"] = [f"{col} (scaled)" for col in X.columns]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=(1 - train_size), random_state=42)

        # Run Logistic Regression
        print(Fore.CYAN + "\nRunning Logistic Regression..." + Style.RESET_ALL)
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        log_reg_preds = log_reg.predict(X_test)
        log_reg_probs = log_reg.predict_proba(X_test)[:, 1]

        print_classification_report_with_confusion_matrix(y_test, log_reg_preds, "Logistic Regression")
        generate_roc_curve(y_test, log_reg_probs, model_name="Logistic Regression")

        # Run Random Forest
        print(Fore.CYAN + "\nRunning Random Forest..." + Style.RESET_ALL)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_probs = rf.predict_proba(X_test)[:, 1]

        print_classification_report_with_confusion_matrix(y_test, rf_preds, "Random Forest")
        generate_roc_curve(y_test, rf_probs, model_name="Random Forest")

        # Log variable details
        print(Fore.GREEN + "\nVariable Log Summary:" + Style.RESET_ALL)
        print(f"Variables Kept: {variables_log['Kept']}")
        print(f"Variables Excluded:")
        for exclusion in variables_log["Excluded"]:
            print(f"  - {exclusion['Variable']}: {exclusion['Reason']}")

        # Visualize variables and their data types
        print(Fore.GREEN + "\nSummary of Variables and Data Types:" + Style.RESET_ALL)
        print(data_types.to_string(index=False))
    
    def filter_significant_features_with_reasons(X, y):
        """Filters features based on logistic regression p-values and logs reasons for exclusion."""
        X_with_const = sm.add_constant(X)
        logit_model = sm.Logit(y, X_with_const).fit(disp=False)
        significant_vars = logit_model.pvalues[logit_model.pvalues < 0.05].index.tolist()
        excluded_vars = [
            {"Variable": var, "Reason": "p-value >= 0.05 (not statistically significant)"}
            for var in X.columns if var not in significant_vars
        ]
        return X[significant_vars], excluded_vars
    
    def remove_multicollinear_features_with_reasons(X):
        """Removes features with high multicollinearity (VIF > 5) and logs reasons for exclusion."""
        vif_data = pd.DataFrame({'Feature': X.columns, 'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
        excluded_vars = [
            {"Variable": row['Feature'], "Reason": f"VIF = {row['VIF']} (high multicollinearity)"}
            for _, row in vif_data.iterrows() if row['VIF'] > 5
        ]
        filtered_X = X.drop(columns=[exclusion['Variable'] for exclusion in excluded_vars])
        return filtered_X, excluded_vars

    def print_classification_report_with_confusion_matrix(y_true, y_pred, model_name):
        """Print classification metrics and visualize the confusion matrix."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(Fore.CYAN + f"\n{model_name} Metrics:" + Style.RESET_ALL)
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix ({model_name})")
        plt.show()

    def visualize_significant_variables(data, response):
        """
        Visualizes the significance of variables based on logistic regression p-values
        with diverse representations and combined significance table.
        """
        processed_data = preprocess_data_for_model(data)
        X = processed_data.drop(columns=[response, 'CustomerID'])
        y = processed_data[response]

        # Add constant term for regression
        X_with_const = sm.add_constant(X)
        logit_model = sm.Logit(y, X_with_const).fit(disp=False)

        # Extract p-values
        pvalues = logit_model.pvalues[1:]  # Skip the constant term
        pvalues.sort_values(inplace=True)

        # Calculate VIF (optional)
        vif = pd.DataFrame({
            'Feature': X.columns,
            'VIF': [
                variance_inflation_factor(X.values, i) if np.var(X.iloc[:, i]) != 0 else float("inf")
                for i in range(X.shape[1])
            ]
        })
        vif.set_index('Feature', inplace=True)

        # Combine p-values and VIFs into a single table
        significance_table = pd.DataFrame({
            'p-value': logit_model.pvalues[1:],  # Exclude constant
            'Significant': logit_model.pvalues[1:] < 0.05,  # Boolean for p-value < 0.05
        }).join(vif, how='left')

        # Print the logistic regression summary
        print(Fore.CYAN + "\nLogistic Regression Summary:" + Style.RESET_ALL)
        print(logit_model.summary())

        # Print the significance table
        print(Fore.CYAN + "\nCombined Significance Table:" + Style.RESET_ALL)
        print(significance_table)

        # Visualization 1: Bar plot of p-values
        plt.figure(figsize=(10, 6))
        sns.barplot(x=pvalues.values, y=pvalues.index, palette="coolwarm", hue=None, legend=False)
        plt.axvline(x=0.05, color='red', linestyle='--', label='Significance Threshold (p=0.05)')
        plt.title("Variable Significance (p-values from Logistic Regression)")
        plt.xlabel("p-value")
        plt.ylabel("Variable")
        plt.legend(title="Significance")
        plt.grid(alpha=0.3)
        plt.show()

        # Visualization 2: Log-transformed p-values
        plt.figure(figsize=(10, 6))
        sns.barplot(x=-np.log10(pvalues.values), y=pvalues.index, palette="coolwarm", hue=None, legend=False)
        plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='-log10(0.05)')
        plt.title("Variable Significance (-log10(p-values))")
        plt.xlabel("-log10(p-value)")
        plt.ylabel("Variable")
        plt.legend(title="Significance")
        plt.grid(alpha=0.3)
        plt.show()

        print(Fore.CYAN + "Variable significance visualization complete." + Style.RESET_ALL)
    def check_and_handle_multicollinearity(data, response):
        """
        Check multicollinearity using Variance Inflation Factor (VIF) and handle if necessary.
        """
        print(Fore.CYAN + "\nChecking Multicollinearity using VIF..." + Style.RESET_ALL)
        
        # Preprocess the data
        processed_data = preprocess_data_for_model(data)
        X = processed_data.drop(columns=[response, 'CustomerID'])
        
        # Calculate VIF for each feature
        vif = pd.DataFrame()
        vif["Feature"] = X.columns
        vif["VIF"] = [
            variance_inflation_factor(X.values, i) if np.var(X.iloc[:, i]) != 0 else float("inf")
            for i in range(X.shape[1])
        ]
        
        # Print the VIF values
        print(Fore.CYAN + "\nVariance Inflation Factor (VIF) Table:" + Style.RESET_ALL)
        print(vif.to_string(index=False))
        
        # Highlight features with high VIF (typically > 5 or > 10 as thresholds)
        high_vif_features = vif[vif['VIF'] > 5]['Feature'].tolist()
        if high_vif_features:
            print(Fore.YELLOW + "\nFeatures with high VIF (indicating multicollinearity):" + Style.RESET_ALL)
            print(", ".join(high_vif_features))
            
            # Provide the option to remove high VIF features
            print(Fore.CYAN + "\nWould you like to remove these features and re-run the model?" + Style.RESET_ALL)
            choice = input("Enter [yes/no]: ").strip().lower()
            
            if choice == 'yes':
                print(Fore.GREEN + "\nRemoving high VIF features and returning the filtered dataset..." + Style.RESET_ALL)
                X = X.drop(columns=high_vif_features)
            else:
                print(Fore.YELLOW + "\nHigh VIF features retained. Proceeding with all variables." + Style.RESET_ALL)
        else:
            print(Fore.GREEN + "\nNo multicollinearity detected. All VIF values are below the threshold." + Style.RESET_ALL)

        # Plot VIF values
        plt.figure(figsize=(8, 6))
        sns.barplot(x='VIF', y='Feature', data=vif, palette="coolwarm", hue=None, legend=False)  # Explicitly set `hue=None`
        plt.axvline(x=5, color='red', linestyle='--', label='VIF Threshold (5)')
        plt.title("Variance Inflation Factor (VIF) for Features")
        plt.xlabel("VIF Value")
        plt.ylabel("Feature")
        plt.legend(title="Threshold")
        plt.grid(alpha=0.3)
        plt.show()
        
        return X  # Return the filtered dataset if features were removed

    def generate_roc_curve(y_true, y_prob, model_name=None):
        """Generates and plots the ROC curve for a given model."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.legend()
        plt.title("ROC Curve")
        plt.show()

    submenu()