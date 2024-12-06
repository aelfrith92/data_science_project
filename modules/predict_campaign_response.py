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

    # Handle missing or infinite values for numeric datetime-derived features
    numeric_features = ['customer_tenure_days', 'recency_days']
    for feature in numeric_features:
        if feature in data.columns:
            # Replace infinities with NaN
            inf_count = data[feature].isin([float('inf'), float('-inf')]).sum()
            if inf_count > 0:
                data[feature].replace([float('inf'), float('-inf')], pd.NA, inplace=True)
                log_messages.append(f"Replaced {inf_count} infinite values in '{feature}' with NaN.")

            # Fill missing values
            nan_count = data[feature].isna().sum()
            if nan_count > 0:
                median_value = data[feature].median()  # Using median to handle missing values
                data[feature].fillna(median_value, inplace=True)
                log_messages.append(f"Filled {nan_count} missing values in '{feature}' with median value {median_value}.")

    # Scale numeric datetime-derived features
    scaler = StandardScaler()
    for feature in numeric_features:
        if feature in data.columns:
            data[feature] = scaler.fit_transform(data[[feature]])
            log_messages.append(f"Scaled numeric feature '{feature}' using StandardScaler.")

    # Handle categorical variables
    categorical_fields = ['loyalty', 'nps', 'recency_bin', 'total_sales_bin', 'total_quantity_bin',
                          'avg_order_value_bin', 'customer_tenure_bin', 'order_frequency_bin']

    for field in categorical_fields:
        if field in data.columns:
            # Ensure the field is categorical
            if not pd.api.types.is_categorical_dtype(data[field]):
                data[field] = pd.Categorical(data[field])
                log_messages.append(f"Converted '{field}' to a categorical data type.")

            # Validate and add missing categories
            if field in ['loyalty', 'nps']:
                if 0 not in data[field].cat.categories:
                    data[field] = data[field].cat.add_categories([0])
                    log_messages.append(f"Added missing category '0' to '{field}'.")

            if field.endswith('_bin'):
                if 'Unknown' not in data[field].cat.categories:
                    data[field] = data[field].cat.add_categories(['Unknown'])
                    log_messages.append(f"Added missing category 'Unknown' to '{field}'.")

            # Fill missing values with default categories
            default_value = 0 if field in ['loyalty', 'nps'] else 'Unknown'
            if data[field].isna().any():
                data[field] = data[field].fillna(default_value)
                log_messages.append(f"Filled missing values in '{field}' with '{default_value}'.")

    # Convert categorical fields to numeric (ordinal codes)
    for field in categorical_fields:
        if field in data.columns:
            if field in ['loyalty', 'nps']:
                data[field] = data[field].astype(int)  # Convert to numeric (integer)
                log_messages.append(f"Converted categorical field '{field}' to integer codes.")
            else:
                data[field] = data[field].cat.codes  # Convert binned fields to numeric codes
                log_messages.append(f"Converted binned categorical field '{field}' to numeric codes.")

    # Replace infinities in the entire dataset
    inf_count = data.isin([float('inf'), float('-inf')]).sum().sum()
    if inf_count > 0:
        data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        log_messages.append(f"Replaced {inf_count} infinite values in the dataset with NaN.")

    # Fill NaN values explicitly for remaining columns
    nan_count_before = data.isna().sum().sum()
    data = data.fillna(0)
    nan_count_after = data.isna().sum().sum()
    log_messages.append(f"Filled {nan_count_before - nan_count_after} remaining NaN values with 0.")

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
    Includes preprocessing, multicollinearity checks, feature significance analysis, and model execution.
    """
    print(Fore.CYAN + "\nPredicting Customer Response..." + Style.RESET_ALL)

    def submenu():
        """Submenu for configuring model parameters and running actions."""
        significant_vars = None
        multicollinear_vars = None

        while True:
            print(Fore.CYAN + "\nCustomer Response Prediction Submenu:" + Style.RESET_ALL)
            print("1. Preprocess Data")
            print("2. Check Multicollinearity and Remove Variables")
            print("3. Explore Feature Significance and Remove Variables")
            print("4. Run Models (with the variables left)")
            print("0. Return to Main Menu")

            choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)

            if choice == '1':
                processed_data = preprocess_data_for_model(data)
                print(Fore.GREEN + "Data preprocessing completed. Ready for modeling." + Style.RESET_ALL)
            elif choice == '2':
                processed_data = preprocess_data_for_model(data)
                multicollinear_vars = check_and_handle_multicollinearity(processed_data, response)
            elif choice == '3':
                if multicollinear_vars is None:
                    print(Fore.RED + "Please run multicollinearity checks first (Option 2)." + Style.RESET_ALL)
                else:
                    significant_vars = explore_feature_significance(multicollinear_vars, response)
            elif choice == '4':
                if significant_vars is None:
                    print(Fore.RED + "Please process features through Options 2 and 3 first." + Style.RESET_ALL)
                else:
                    run_models(processed_data, significant_vars, response)
            elif choice == '0':
                print(Fore.CYAN + "Returning to Main Menu..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Invalid choice! Please try again." + Style.RESET_ALL)

    def check_and_handle_multicollinearity(data, response):
        """
        Iteratively calculate VIF and remove features with the highest VIF > 5.
        Ensure the response column is retained in the returned dataset.
        """
        print(Fore.CYAN + "\nChecking Multicollinearity using VIF..." + Style.RESET_ALL)

        # Handle missing 'CustomerID'
        if 'CustomerID' not in data.columns:
            print(Fore.YELLOW + "'CustomerID' not found in dataset. Proceeding without it." + Style.RESET_ALL)

        # Drop only columns that exist
        columns_to_drop = [col for col in [response, 'CustomerID'] if col in data.columns]
        X = data.drop(columns=columns_to_drop)
        y = data[response]

        iteration = 1
        while True:
            vif = pd.DataFrame({
                "Feature": X.columns,
                "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            })
            print(Fore.CYAN + f"\nVIF Calculation - Iteration {iteration}:" + Style.RESET_ALL)
            print(vif.to_string(index=False))

            high_vif = vif[vif["VIF"] > 5]
            if high_vif.empty:
                print(Fore.GREEN + "\nNo multicollinearity detected. All VIF values are ≤ 5." + Style.RESET_ALL)
                break

            to_remove = high_vif.sort_values("VIF", ascending=False).iloc[0]["Feature"]
            X = X.drop(columns=[to_remove])

            print(Fore.YELLOW + f"\nRemoving feature '{to_remove}' with VIF = {high_vif.loc[high_vif['Feature'] == to_remove, 'VIF'].values[0]}." + Style.RESET_ALL)

            iteration += 1

        # Reattach the response column to the filtered dataset
        filtered_data = pd.concat([X, y], axis=1)

        # Final VIF visualization
        plt.figure(figsize=(8, 6))
        sns.barplot(x="VIF", y="Feature", data=vif, palette="coolwarm", order=vif.sort_values("VIF", ascending=False)["Feature"])
        plt.axvline(x=5, color='red', linestyle='--', label='VIF Threshold (5)')
        plt.title("Final Variance Inflation Factor (VIF) After Multicollinearity Check")
        plt.xlabel("VIF Value")
        plt.ylabel("Feature")
        plt.legend(title="Threshold")
        plt.grid(alpha=0.3)
        plt.show()

        print(Fore.GREEN + f"\nFinal Features After Handling Multicollinearity: {list(X.columns)}" + Style.RESET_ALL)
        return filtered_data  # Include response

    def explore_feature_significance(data, response):
        """
        Visualizes the significance of variables based on logistic regression p-values.
        Includes log-10 scale visualization only.
        """
        processed_data = preprocess_data_for_model(data)

        # Dynamically handle 'CustomerID' and 'response'
        columns_to_drop = [col for col in [response, 'CustomerID'] if col in processed_data.columns]
        X = processed_data.drop(columns=columns_to_drop)
        y = processed_data[response]

        # Add constant term for regression
        X_with_const = sm.add_constant(X)
        logit_model = sm.Logit(y, X_with_const).fit(disp=False)

        # Extract p-values
        pvalues = logit_model.pvalues[1:]  # Skip the constant term
        pvalues.sort_values(inplace=True)

        # Log-10 scale visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x=-np.log10(pvalues.values), y=pvalues.index, palette="coolwarm", hue=None, legend=False)
        plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='-log10(0.05)')
        plt.title("Variable Significance (-log10(p-values))")
        plt.xlabel("-log10(p-value)")
        plt.ylabel("Variable")
        plt.legend(title="Significance Threshold")
        plt.grid(alpha=0.3)
        plt.show()

        # Log the logistic regression summary
        print(Fore.CYAN + "\nLogistic Regression Summary:" + Style.RESET_ALL)
        print(logit_model.summary())

        # Return significant variables
        return [
            var for var in logit_model.pvalues.index
            if var != 'const' and logit_model.pvalues[var] < 0.05
        ]

    def run_models(data, significant_vars, response):
        """
        Execute Logistic Regression or Random Forest models with the significant variables.
        """
        print(Fore.CYAN + "\nRunning Models..." + Style.RESET_ALL)

        # Filter the dataset to include only the significant variables and the response
        X = data[significant_vars]
        y = data[response]

        # Train-test split
        train_size = float(input(Fore.YELLOW + "Enter train/test split ratio (e.g., 0.7 for 70% train): " + Style.RESET_ALL))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_size), random_state=42)

        print(Fore.CYAN + "\nSelect Model:" + Style.RESET_ALL)
        print("1. Logistic Regression")
        print("2. Random Forest")
        model_choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)

        variables_log = {"Variables Used": list(X.columns)}

        # Logistic Regression
        if model_choice == '1':
            log_reg = LogisticRegression(max_iter=1000)
            log_reg.fit(X_train, y_train)

            # Train Data Evaluation
            train_preds = log_reg.predict(X_train)
            train_probs = log_reg.predict_proba(X_train)[:, 1]
            train_metrics = evaluate_model(y_train, train_preds, train_probs, "Logistic Regression", dataset_type="Train")

            # Test Data Evaluation
            test_preds = log_reg.predict(X_test)
            test_probs = log_reg.predict_proba(X_test)[:, 1]
            evaluate_model(y_test, test_preds, test_probs, "Logistic Regression", train_metrics=train_metrics, dataset_type="Test")

        # Random Forest
        elif model_choice == '2':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            # Train Data Evaluation
            train_preds = rf.predict(X_train)
            train_probs = rf.predict_proba(X_train)[:, 1]
            train_metrics = evaluate_model(y_train, train_preds, train_probs, "Random Forest", dataset_type="Train")

            # Test Data Evaluation
            test_preds = rf.predict(X_test)
            test_probs = rf.predict_proba(X_test)[:, 1]
            evaluate_model(y_test, test_preds, test_probs, "Random Forest", train_metrics=train_metrics, dataset_type="Test")
        else:
            print(Fore.RED + "Invalid choice. Returning to submenu." + Style.RESET_ALL)

        print(Fore.GREEN + f"\nVariables Used for the Model: {variables_log['Variables Used']}" + Style.RESET_ALL)

    def evaluate_model(y_true, y_pred, y_prob, model_name, train_metrics=None, dataset_type="Test"):
        """
        Evaluate models and log metrics, including confusion matrix, thresholds, and AUC.
        Handles separate logs for train and test datasets.
        """
        print(Fore.CYAN + f"\n{model_name} Evaluation on {dataset_type} Data:" + Style.RESET_ALL)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2

        # Best Threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[optimal_idx]

        # Log Metrics
        print(f"Accuracy: {accuracy:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Balanced Accuracy: {balanced_accuracy:.2f}")
        print(f"Best Threshold: {best_threshold:.2f}")

        # Overfitting Assessment (for test data only)
        if train_metrics is not None and dataset_type == "Test":
            print(Fore.YELLOW + "\nOverfitting Assessment:" + Style.RESET_ALL)
            print(f"Train-Test Accuracy Gap: {train_metrics['accuracy'] - accuracy:.2f}")
            print(f"Train-Test Balanced Accuracy Gap: {train_metrics['balanced_accuracy'] - balanced_accuracy:.2f}")
            if abs(train_metrics['accuracy'] - accuracy) > 0.1:
                print(Fore.RED + "Warning: Potential Overfitting Detected (Accuracy Gap > 0.1)." + Style.RESET_ALL)

        # Save Confusion Matrix Plot
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f"Confusion Matrix ({model_name}, {dataset_type})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"./assets/imgs/confusion_matrix_{model_name}_{dataset_type.lower()}.png")
        plt.show()

        # Save ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f"ROC Curve ({model_name}, {dataset_type})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f"./assets/imgs/roc_curve_{model_name}_{dataset_type.lower()}.png")
        plt.show()

        # Return Metrics
        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "best_threshold": best_threshold,
        }

    submenu()
