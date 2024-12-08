from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from graphviz import Source
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

global_logs = {
    "preprocessing": {},
    "multicollinearity": {},
    "significance": {},
    "models": {
        "baseline": {},
        "recoded": {},
        "outlier_removed": {},
        "outlier_recoded_scaled": {}
    }
}

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
            inf_count = data[feature].isin([float('inf'), float('-inf')]).sum()
            if inf_count > 0:
                data[feature].replace([float('inf'), float('-inf')], pd.NA, inplace=True)
                log_messages.append(f"Replaced {inf_count} infinite values in '{feature}' with NaN.")
            nan_count = data[feature].isna().sum()
            if nan_count > 0:
                median_value = data[feature].median()
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
            if not pd.api.types.is_categorical_dtype(data[field]):
                data[field] = pd.Categorical(data[field])
                log_messages.append(f"Converted '{field}' to a categorical data type.")
            if field in ['loyalty', 'nps'] and 0 not in data[field].cat.categories:
                data[field] = data[field].cat.add_categories([0])
                log_messages.append(f"Added missing category '0' to '{field}'.")
            if field.endswith('_bin') and 'Unknown' not in data[field].cat.categories:
                data[field] = data[field].cat.add_categories(['Unknown'])
                log_messages.append(f"Added missing category 'Unknown' to '{field}'.")
            default_value = 0 if field in ['loyalty', 'nps'] else 'Unknown'
            if data[field].isna().any():
                data[field] = data[field].fillna(default_value)
                log_messages.append(f"Filled missing values in '{field}' with '{default_value}'.")
            if field in ['loyalty', 'nps']:
                data[field] = data[field].astype(int)
                log_messages.append(f"Converted categorical field '{field}' to integer codes.")
            else:
                data[field] = data[field].cat.codes
                log_messages.append(f"Converted binned categorical field '{field}' to numeric codes.")

    # Replace infinities and handle remaining NaN values
    inf_count = data.isin([float('inf'), float('-inf')]).sum().sum()
    if inf_count > 0:
        data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        log_messages.append(f"Replaced {inf_count} infinite values in the dataset with NaN.")
    nan_count_before = data.isna().sum().sum()
    data = data.fillna(0)
    nan_count_after = data.isna().sum().sum()
    log_messages.append(f"Filled {nan_count_before - nan_count_after} remaining NaN values with 0.")

    # Log preprocessing steps
    global_logs["preprocessing"]["steps"] = log_messages

    # Print detailed log
    print(Fore.CYAN + "\nPreprocessing Log:" + Style.RESET_ALL)
    for message in log_messages:
        print(Fore.GREEN + "✔ " + message + Style.RESET_ALL)

    return data

def predict_customer_response(data, response='response'):
    """
    Predict customer response using Logistic Regression and Random Forest models.
    Includes preprocessing, multicollinearity checks, feature significance analysis, and model execution.
    """
    print(Fore.CYAN + "\nPredicting Customer Response..." + Style.RESET_ALL)

    def submenu():
        """Submenu for configuring model parameters and running actions."""

        while True:
            print(Fore.CYAN + "\nCustomer Response Prediction Submenu:" + Style.RESET_ALL)
            print("1. Preprocess Data")
            print("2. Run Models (Baseline)")
            print("3. Run Models (with recoded variables and scaling)")
            print("4. Run Models (with outlier removal)")
            print("5. Run Models (with outlier removal, recoded variables, and scaling)")
            print("6. Summarize Results of All Models")
            print("7. Naive Bayes and Decision Trees")
            print("0. Return to Main Menu")

            choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)

            if choice == '1':
                processed_data = preprocess_data_for_model(data)
                print(Fore.GREEN + "Data preprocessing completed. Ready for modeling." + Style.RESET_ALL)
            elif choice == '2':
                # Preprocess the data
                processed_data = preprocess_data_for_model(data)

                # Log the columns
                print(Fore.CYAN + "\nColumns After Preprocessing:" + Style.RESET_ALL)
                print(Fore.GREEN + ", ".join(processed_data.columns) + Style.RESET_ALL)

                # Handle multicollinearity
                filtered_data = check_and_handle_multicollinearity(processed_data, response)

                # Log the columns
                print(Fore.CYAN + "\nColumns After Multicollinearity Check:" + Style.RESET_ALL)
                print(Fore.GREEN + ", ".join(filtered_data.columns) + Style.RESET_ALL)

                # Perform significance testing
                significant_vars = explore_feature_significance(filtered_data, response)

                # Log the significant variables
                print(Fore.CYAN + "\nSignificant Variables:" + Style.RESET_ALL)
                print(Fore.GREEN + ", ".join(significant_vars) + Style.RESET_ALL)

                if significant_vars:
                    configuration_key = "baseline"
                    run_models(filtered_data, significant_vars, response, configuration_key)
                else:
                    print(Fore.RED + "No significant variables found for baseline model." + Style.RESET_ALL)
            elif choice == '3':
                run_models_with_recoded_variables(processed_data, response=response)
            elif choice == '4':
                run_models_with_outlier_removal(processed_data, response=response)
            elif choice == '5':
                no_outliers_and_recoded_scaled_variables(processed_data, response=response)
            elif choice == '6':
                summarize_results()
            elif choice == '7':
                run_model_evaluation_submenu(processed_data, response=response)
            elif choice == '0':
                print(Fore.CYAN + "Returning to Main Menu..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Invalid choice! Please try again." + Style.RESET_ALL)

    def check_and_handle_multicollinearity(data, response):
        """
        Check multicollinearity using VIF and iteratively remove variables with VIF > 5.
        Modular function for repeated use across models.
        """
        print(Fore.CYAN + "\nPerforming Multicollinearity Check..." + Style.RESET_ALL)

        log_messages = []
        X = data.select_dtypes(include=[np.number]).drop(columns=[response], errors='ignore')

        # Handle missing and non-finite values
        if not np.isfinite(X).all().all():
            non_finite_count = (~np.isfinite(X)).sum().sum()
            print(Fore.YELLOW + f"Found {non_finite_count} non-finite values. Replacing with zeros." + Style.RESET_ALL)
            X = X.fillna(0).replace([np.inf, -np.inf], 0)

        vif_data = []
        iteration = 1

        while True:
            vif = pd.DataFrame({
                "Feature": X.columns,
                "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            })

            vif_data.append(vif.copy())
            print(Fore.CYAN + f"\nVIF Calculation - Iteration {iteration}:" + Style.RESET_ALL)
            print(vif.to_string(index=False))

            high_vif = vif[vif["VIF"] > 5]
            if high_vif.empty:
                log_messages.append("No multicollinearity detected. All VIF values are ≤ 5.")
                break

            to_remove = high_vif.sort_values("VIF", ascending=False).iloc[0]["Feature"]
            X = X.drop(columns=[to_remove])
            log_messages.append(f"Removed feature '{to_remove}' with VIF = {high_vif.iloc[0]['VIF']:.2f}.")
            iteration += 1

        global_logs["multicollinearity"]["vif"] = vif_data[-1].to_dict("records")
        global_logs["multicollinearity"]["steps"] = log_messages

        # Final VIF Plot
        final_vif = vif_data[-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x="VIF", y="Feature", data=final_vif.sort_values("VIF", ascending=False), palette="coolwarm")
        plt.axvline(x=5, color='red', linestyle='--', label='VIF Threshold (5)')
        plt.title("Final Variance Inflation Factor (VIF) After Multicollinearity Check")
        plt.xlabel("VIF Value")
        plt.ylabel("Feature")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        print(Fore.CYAN + "\nMulticollinearity Log:" + Style.RESET_ALL)
        for log in log_messages:
            print(Fore.GREEN + "✔ " + log + Style.RESET_ALL)

        # Reattach the response column
        filtered_data = pd.concat([X, data[response]], axis=1)
        return filtered_data

    def explore_feature_significance(data, response):
        """
        Perform feature significance analysis using logistic regression.
        Modular function for repeated use across models.
        """
        print(Fore.CYAN + "\nExploring Feature Significance..." + Style.RESET_ALL)
        X = data.drop(columns=[response]) if response in data.columns else data
        y = data[response]

        # Add constant term for regression
        X_with_const = sm.add_constant(X)
        logit_model = sm.Logit(y, X_with_const).fit(disp=False)

        # Extract p-values
        pvalues = logit_model.pvalues[1:]  # Skip the constant term
        pvalues.sort_values(inplace=True)

        # Log-10 scale visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x=-np.log10(pvalues.values), y=pvalues.index, palette="coolwarm")
        plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='-log10(0.05)')
        plt.title("Feature Significance (-log10(p-values))")
        plt.xlabel("-log10(p-value)")
        plt.ylabel("Variable")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        # Log significant variables
        significant_vars = [
            var for var in logit_model.pvalues.index
            if var != 'const' and logit_model.pvalues[var] < 0.05
        ]
        global_logs["significance"]["variables"] = significant_vars

        # Print logistic regression summary
        print(Fore.CYAN + "\nLogistic Regression Summary:" + Style.RESET_ALL)
        print(logit_model.summary())
        return significant_vars

    def run_models(data, significant_vars, response, configuration_key):
        print(Fore.CYAN + "\nValidating Significant Variables Against Data Columns..." + Style.RESET_ALL)
        missing_vars = [var for var in significant_vars if var not in data.columns]
        if missing_vars:
            print(Fore.RED + f"Error: The following significant variables are missing from the dataset: {missing_vars}" + Style.RESET_ALL)
            print(Fore.CYAN + "Available Columns in Data:" + Style.RESET_ALL)
            print(Fore.GREEN + ", ".join(data.columns) + Style.RESET_ALL)
            return  # Exit the function to prevent further errors.

        # Proceed only with valid significant variables
        significant_vars = [var for var in significant_vars if var in data.columns]
        X = data[significant_vars]
        y = data[response]

        # Train-test split with improved validation
        while True:
            try:
                train_size = float(input(Fore.YELLOW + "Enter train/test split ratio (e.g., 0.7 for 70% train): " + Style.RESET_ALL))
                if 0.0 < train_size < 1.0:
                    break
                else:
                    print(Fore.RED + "Invalid input. Please enter a value between 0.0 and 1.0." + Style.RESET_ALL)
            except ValueError:
                print(Fore.RED + "Invalid input. Please enter a numeric value." + Style.RESET_ALL)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_size), random_state=42)

        print(Fore.CYAN + "\nSelect Model:" + Style.RESET_ALL)
        print("1. Logistic Regression")
        print("2. Random Forest")
        model_choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)

        if model_choice == '1':
            model = LogisticRegression(max_iter=1000)
            model_name = "Logistic Regression"
        elif model_choice == '2':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model_name = "Random Forest"
        else:
            print(Fore.RED + "Invalid choice. Returning to submenu." + Style.RESET_ALL)
            return

        # Train the selected model
        model.fit(X_train, y_train)

        # Evaluate model performance on train and test datasets
        train_metrics = evaluate_model(y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1], model_name, dataset_type="Train")
        test_metrics = evaluate_model(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1], model_name, train_metrics=train_metrics, dataset_type="Test")

        # Initialize the configuration key in global_logs if not already initialized
        if global_logs["models"][configuration_key] is None:
            global_logs["models"][configuration_key] = {}

        # Log results in global_logs
        model_key = f"{model_name.lower().replace(' ', '_')}_results"
        global_logs["models"][configuration_key][model_key] = {
            "train_accuracy": train_metrics.get("accuracy"),
            "test_accuracy": test_metrics.get("accuracy"),
            "variables_used": significant_vars,
            "overfitting_gap": train_metrics.get("accuracy") - test_metrics.get("accuracy"),
            "balanced_accuracy_gap": train_metrics.get("balanced_accuracy") - test_metrics.get("balanced_accuracy"),
        }
        print(Fore.GREEN + f"\n{model_name} Model Results logged: {global_logs['models'][configuration_key][model_key]}" + Style.RESET_ALL)

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

    def run_models_with_recoded_variables(data, response):
        """
        Run models with recoded variables and scaling.
        Includes multicollinearity checks and significance reassessment.
        """
        print(Fore.CYAN + "\nRunning Models with Recoded Variables..." + Style.RESET_ALL)

        # Preprocessing
        data = preprocess_data_for_model(data)

        # Add Recoded Variables
        if 'num_orders' in data.columns and 'recency_days' in data.columns:
            data['recency_ratio'] = data['recency_days'] / (data['num_orders'] + 1e-5)
            data['complaint_ratio'] = data['n_comp'] / (data['num_orders'] + 1e-5)
            print(Fore.GREEN + "✔ Added 'recency_ratio' and 'complaint_ratio' to the dataset." + Style.RESET_ALL)

        # Scale Features
        scaler = StandardScaler()
        for feature in ['recency_ratio', 'complaint_ratio']:
            if feature in data.columns:
                data[feature] = scaler.fit_transform(data[[feature]])
                print(Fore.GREEN + f"✔ Scaled feature '{feature}' using StandardScaler." + Style.RESET_ALL)

        # Multicollinearity Check
        data = check_and_handle_multicollinearity(data, response)

        # Significance Analysis
        significant_vars = explore_feature_significance(data, response)

        if not significant_vars:
            print(Fore.RED + "No significant variables found after multicollinearity checks." + Style.RESET_ALL)
            return

        configuration_key = "recoded"
        # Run Models
        run_models(data, significant_vars, response, configuration_key)

    def run_models_with_outlier_removal(data, response):
        """
        Run models with outlier removal, multicollinearity checks, and significance reassessment.
        """
        print(Fore.CYAN + "\nRunning Models with Outlier Removal..." + Style.RESET_ALL)

        # Preprocessing
        data = preprocess_data_for_model(data)

        # Remove Outliers
        numerical_fields = data.select_dtypes(include=['float64', 'int64']).columns
        for field in numerical_fields:
            if field == response:
                continue
            Q1 = data[field].quantile(0.25)
            Q3 = data[field].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data[field] < lower_bound) | (data[field] > upper_bound)
            data = data[~outliers]
            print(Fore.GREEN + f"Removed outliers from '{field}'." + Style.RESET_ALL)

        # Multicollinearity Check
        data = check_and_handle_multicollinearity(data, response)

        # Significance Analysis
        significant_vars = explore_feature_significance(data, response)

        if not significant_vars:
            print(Fore.RED + "No significant variables found after multicollinearity checks." + Style.RESET_ALL)
            return

        configuration_key = "outlier_removed"
        # Run Models
        run_models(data, significant_vars, response, configuration_key)

    def no_outliers_and_recoded_scaled_variables(data, response='response'):
        """
        Run models with outlier removal, recoded variables, and scaling.
        Includes multicollinearity checks and significance reassessment.
        Logs results into `global_logs`.
        """
        print(Fore.CYAN + "\nRunning Models with Outlier Removal, Recoded Variables, and Scaling..." + Style.RESET_ALL)

        # Step 1: Preprocessing
        processed_data = preprocess_data_for_model(data)

        # Step 2: Adding Recoded Variables
        if 'num_orders' in processed_data.columns and 'recency_days' in processed_data.columns:
            processed_data['recency_ratio'] = processed_data['recency_days'] / (processed_data['num_orders'] + 1e-5)
            processed_data['complaint_ratio'] = processed_data['n_comp'] / (processed_data['num_orders'] + 1e-5)
            print(Fore.GREEN + "✔ Added 'recency_ratio' and 'complaint_ratio' to the dataset." + Style.RESET_ALL)

        # Step 3: Scaling Recoded Features
        scaler = StandardScaler()
        for feature in ['recency_ratio', 'complaint_ratio']:
            if feature in processed_data.columns:
                processed_data[feature] = scaler.fit_transform(processed_data[[feature]])
                print(Fore.GREEN + f"✔ Scaled numeric feature '{feature}' using StandardScaler." + Style.RESET_ALL)

        # Step 4: Detect and Remove Outliers
        numerical_fields = processed_data.select_dtypes(include=['float64', 'int64']).columns
        outlier_logs = []

        for field in numerical_fields:
            if field == response:
                continue  # Skip the response field
            Q1 = processed_data[field].quantile(0.25)
            Q3 = processed_data[field].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Detect and remove outliers
            outliers = ((processed_data[field] < lower_bound) | (processed_data[field] > upper_bound))
            outlier_count = outliers.sum()
            if outlier_count > 0:
                processed_data = processed_data[~outliers]
                outlier_logs.append(f"Removed {outlier_count} outliers from '{field}'. Remaining rows: {len(processed_data)}")

        # Log outlier removal results
        if outlier_logs:
            print(Fore.GREEN + "\nOutlier Removal Summary:" + Style.RESET_ALL)
            for log in outlier_logs:
                print(Fore.GREEN + "✔ " + log + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "No outliers detected for numerical fields." + Style.RESET_ALL)

        # Step 5: Multicollinearity Check
        print(Fore.CYAN + "\nChecking Multicollinearity..." + Style.RESET_ALL)
        filtered_data = check_and_handle_multicollinearity(processed_data, response)

        # Step 6: Reassess Significance
        print(Fore.CYAN + "\nPerforming Significance Testing..." + Style.RESET_ALL)
        significant_vars = explore_feature_significance(filtered_data, response)

        # Step 7: Run Models
        if significant_vars:
            print(Fore.CYAN + "\nRunning Models on Cleaned Dataset with Recoded Variables..." + Style.RESET_ALL)
            configuration_key = "outlier_recoded_scaled"
            # Run Models
            run_models(filtered_data, significant_vars, response, configuration_key)
        else:
            print(Fore.RED + "No significant variables identified after outlier removal and multicollinearity checks." + Style.RESET_ALL)
            return

    def summarize_results():
        """
        Summarizes preprocessing, multicollinearity checks, significance analysis, and model performance.
        Visualizes aggregated performance metrics for all models and configurations.
        """
        print(Fore.CYAN + "\nSummary of All Model Runs:" + Style.RESET_ALL)

        # Step 1: Preprocessing Overview
        print(Fore.CYAN + "\nPreprocessing Summary:" + Style.RESET_ALL)
        preprocessing_steps = global_logs.get("preprocessing", {}).get("steps", [])
        if preprocessing_steps:
            for step in preprocessing_steps:
                print(Fore.GREEN + f"✔ {step}" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "No preprocessing steps logged." + Style.RESET_ALL)

        # Step 2: Multicollinearity Results
        print(Fore.CYAN + "\nMulticollinearity Checks:" + Style.RESET_ALL)
        vif_results = global_logs.get("multicollinearity", {}).get("vif")
        if vif_results:
            print(Fore.GREEN + "✔ Final VIF Values:" + Style.RESET_ALL)
            for record in vif_results:
                print(Fore.GREEN + f"   {record['Feature']}: VIF = {record['VIF']:.2f}" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "No multicollinearity results logged." + Style.RESET_ALL)

        # Step 3: Model Performance Comparison
        print(Fore.CYAN + "\nModel Performance Comparison:" + Style.RESET_ALL)
        model_results = global_logs.get("models", {})
        all_metrics = []

        for config, models in model_results.items():
            if models:  # Ensure results exist for this configuration
                for model_name, metrics in models.items():
                    try:
                        print(Fore.GREEN + f"\n{config.capitalize()} - {model_name.replace('_', ' ').capitalize()} Model Results:" + Style.RESET_ALL)
                        print(f"Train Accuracy: {metrics['train_accuracy']:.2f}")
                        print(f"Test Accuracy: {metrics['test_accuracy']:.2f}")
                        print(f"Overfitting Gap: {metrics['overfitting_gap']:.2f}")
                        print(f"Balanced Accuracy Gap: {metrics['balanced_accuracy_gap']:.2f}")
                        print(f"Variables Used: {', '.join(metrics['variables_used'])}")

                        # Store metrics for aggregated visualization
                        all_metrics.append({
                            "Configuration": config.capitalize(),
                            "Model": model_name.replace('_', ' ').capitalize(),
                            "Train Accuracy": metrics['train_accuracy'],
                            "Test Accuracy": metrics['test_accuracy'],
                            "Overfitting Gap": metrics['overfitting_gap'],
                            "Balanced Accuracy Gap": metrics['balanced_accuracy_gap']
                        })
                    except KeyError as e:
                        print(Fore.RED + f"Missing key in results: {e}" + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + f"No results logged for {config} configuration." + Style.RESET_ALL)

        # Step 4: Generate Aggregated Performance Chart
        if all_metrics:
            generate_aggregated_performance_chart(all_metrics)
        else:
            print(Fore.RED + "No performance metrics available for visualization." + Style.RESET_ALL)

    def generate_aggregated_performance_chart(metrics):
        """
        Generates an aggregated performance chart for all model configurations and metrics.
        """
        # Convert metrics to DataFrame for easy manipulation
        df = pd.DataFrame(metrics)

        # Aggregated Accuracy Comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df,
            x="Configuration",
            y="Test Accuracy",
            hue="Model",
            edgecolor="black",
        )
        plt.title("Test Accuracy Comparison Across Configurations and Models")
        plt.ylabel("Test Accuracy")
        plt.xlabel("Configuration")
        plt.legend(title="Model")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

        # Overfitting Gaps
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df,
            x="Configuration",
            y="Overfitting Gap",
            hue="Model",
            palette="coolwarm",
            edgecolor="black",
        )
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.title("Overfitting Gap Across Configurations and Models")
        plt.ylabel("Overfitting Gap (Train - Test Accuracy)")
        plt.xlabel("Configuration")
        plt.legend(title="Model")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    def preprocess_for_configuration(data, config_key, response="response"):
        """
        Preprocess the data based on the selected configuration.
        Returns the modified dataset.
        """
        processed_data = preprocess_data_for_model(data)

        if config_key == "baseline":
            # Perform multicollinearity and significance checks
            processed_data = check_and_handle_multicollinearity(processed_data, response)
            significant_vars = explore_feature_significance(processed_data, response)
            if significant_vars:
                processed_data = processed_data[significant_vars + [response]]
            return processed_data

        if config_key == "recoded":
            # Add derived features and scale them
            if 'num_orders' in processed_data.columns and 'recency_days' in processed_data.columns:
                processed_data['recency_ratio'] = processed_data['recency_days'] / (processed_data['num_orders'] + 1e-5)
                processed_data['complaint_ratio'] = processed_data['n_comp'] / (processed_data['num_orders'] + 1e-5)

            numerical_fields = ['recency_ratio', 'complaint_ratio']
            scaler = StandardScaler()
            for field in numerical_fields:
                if field in processed_data.columns:
                    processed_data[field] = scaler.fit_transform(processed_data[[field]])

            # Perform multicollinearity and significance checks
            processed_data = check_and_handle_multicollinearity(processed_data, response)
            significant_vars = explore_feature_significance(processed_data, response)
            if significant_vars:
                processed_data = processed_data[significant_vars + [response]]
            return processed_data

        if config_key == "outlier_removed":
            # Remove outliers
            numerical_fields = processed_data.select_dtypes(include=['float64', 'int64']).columns
            for field in numerical_fields:
                if field == response:
                    continue
                Q1 = processed_data[field].quantile(0.25)
                Q3 = processed_data[field].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_data = processed_data[(processed_data[field] >= lower_bound) & (processed_data[field] <= upper_bound)]

            # Perform multicollinearity and significance checks
            processed_data = check_and_handle_multicollinearity(processed_data, response)
            significant_vars = explore_feature_significance(processed_data, response)
            if significant_vars:
                processed_data = processed_data[significant_vars + [response]]
            return processed_data

        if config_key == "outlier_recoded_scaled":
            # Combine outlier removal and recoded logic
            processed_data = preprocess_for_configuration(data, "recoded", response)
            numerical_fields = processed_data.select_dtypes(include=['float64', 'int64']).columns
            for field in numerical_fields:
                if field == response:
                    continue
                Q1 = processed_data[field].quantile(0.25)
                Q3 = processed_data[field].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_data = processed_data[(processed_data[field] >= lower_bound) & (processed_data[field] <= upper_bound)]
            return processed_data

    def run_model_evaluation_submenu(processed_data, response="response"):
        """
        Submenu for evaluating Naïve Bayes and Decision Tree across configurations.
        """
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier

        while True:
            print(Fore.CYAN + "\nEvaluate Naïve Bayes and Decision Tree Models:" + Style.RESET_ALL)
            print("1. Run Naïve Bayes (Baseline)")
            print("2. Run Naïve Bayes (Recoded)")
            print("3. Run Naïve Bayes (Outlier Removed)")
            print("4. Run Naïve Bayes (Outlier Recoded Scaled)")
            print("5. Run Decision Tree (Baseline)")
            print("6. Run Decision Tree (Recoded)")
            print("7. Run Decision Tree (Outlier Removed)")
            print("8. Run Decision Tree (Outlier Recoded Scaled)")
            print("9. Compare AUC for All Models")
            print("0. Return to Main Menu")

            choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)

            if choice == '1':
                evaluate_naive_bayes("baseline", processed_data, response)
            elif choice == '2':
                evaluate_naive_bayes("recoded", processed_data, response)
            elif choice == '3':
                evaluate_naive_bayes("outlier_removed", processed_data, response)
            elif choice == '4':
                evaluate_naive_bayes("outlier_recoded_scaled", processed_data, response)
            elif choice == '5':
                evaluate_decision_tree("baseline", processed_data, response)
            elif choice == '6':
                evaluate_decision_tree("recoded", processed_data, response)
            elif choice == '7':
                evaluate_decision_tree("outlier_removed", processed_data, response)
            elif choice == '8':
                evaluate_decision_tree("outlier_recoded_scaled", processed_data, response)
            elif choice == '9':
                compare_all_models_auc()
            elif choice == '0':
                print(Fore.CYAN + "Returning to Main Menu..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Invalid choice! Please try again." + Style.RESET_ALL)    

    def evaluate_model_performance(y_true, y_pred, y_prob, model_name, dataset_type):
        """
        Evaluate model performance and return key metrics.
        Includes confusion matrix, AUC, and ROC Curve.
        """
        print(Fore.CYAN + f"\nEvaluating {model_name} on {dataset_type} Data..." + Style.RESET_ALL)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)

        # Print Metrics
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall (Sensitivity): {recall:.2f}")
        print(f"Specificity: {specificity:.2f}, F1 Score: {f1:.2f}, AUC: {auc_score:.2f}")

        # Save and Display Confusion Matrix
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f"Confusion Matrix ({model_name}, {dataset_type})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Save and Display ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f"ROC Curve ({model_name}, {dataset_type})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

        # Return Metrics as a Dictionary
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "auc": auc_score,
            "confusion_matrix": cm.tolist()  # Save as a list for logging
        }

    def evaluate_naive_bayes(config_key, data, response="response"):
        print(Fore.CYAN + f"\nRunning Naïve Bayes for {config_key.capitalize()} Configuration..." + Style.RESET_ALL)

        # Preprocess data for the given configuration
        filtered_data = preprocess_for_configuration(data, config_key, response)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            filtered_data.drop(columns=[response]), filtered_data[response], test_size=0.2, random_state=42
        )

        # Train Naïve Bayes
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Evaluate Train Data
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_metrics = evaluate_model_performance(y_train, y_train_pred, y_train_prob, "Naive Bayes", "Train")

        # Evaluate Test Data
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_model_performance(y_test, y_test_pred, y_test_prob, "Naive Bayes", "Test")

        # Log Results
        global_logs["models"][config_key]["naive_bayes_results"] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }

    def evaluate_decision_tree(config_key, data, response="response"):
        """
        Train and evaluate Decision Tree for the given configuration.

        Parameters:
        - config_key: Configuration key (e.g., 'baseline', 'recoded')
        - data: Input dataset
        - response: Target variable
        """
        print(Fore.CYAN + f"\nRunning Decision Tree for {config_key.capitalize()} Configuration..." + Style.RESET_ALL)

        # Preprocess data for the given configuration
        filtered_data = preprocess_for_configuration(data, config_key, response)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            filtered_data.drop(columns=[response]), filtered_data[response], test_size=0.2, random_state=42
        )

        # Train Decision Tree
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        visualize_decision_tree(
            model=model,
            feature_names=X_train.columns,
            class_names=["No", "Yes"],
            config_key=config_key,  # Pass the configuration key
            file_name="decision_tree_visualization"
        )

        # Evaluate Train Data
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_metrics = evaluate_model_performance(y_train, y_train_pred, y_train_prob, "Decision Tree", "Train")

        # Evaluate Test Data
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_model_performance(y_test, y_test_pred, y_test_prob, "Decision Tree", "Test")

        # Log Results
        if config_key not in global_logs["models"]:
            global_logs["models"][config_key] = {}
        global_logs["models"][config_key]["decision_tree_results"] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }

    def visualize_decision_tree(model, feature_names, class_names, config_key=None, file_name="tree_visualization"):
        """
        Visualizes the decision tree and saves it as an image.
        """
        from sklearn.tree import export_graphviz
        from graphviz import Source

        # Append `config_key` to the file name if provided
        if config_key:
            file_name = f"{file_name}_{config_key}"

        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph = Source(dot_data)
        graph.render(file_name, format="png", cleanup=True)
        print(f"Decision tree visualization saved as {file_name}.png")

    def compare_all_models_auc():
        """
        Compare AUC for all models across configurations and visualize the results.
        """
        print(Fore.CYAN + "\nComparing AUC for All Models..." + Style.RESET_ALL)

        metrics = []

        # Iterate through all configurations and model results
        for config, models in global_logs["models"].items():
            for model_name, results in models.items():
                try:
                    # Extract train and test AUC if available
                    train_auc = results.get("train_metrics", {}).get("auc", None)
                    test_auc = results.get("test_metrics", {}).get("auc", None)

                    if train_auc is not None and test_auc is not None:
                        metrics.append({
                            "Configuration": config.capitalize(),
                            "Model": model_name.replace("_", " ").capitalize(),
                            "Train AUC": train_auc,
                            "Test AUC": test_auc,
                        })
                    else:
                        print(Fore.YELLOW + f"Skipping {model_name} for {config} due to missing AUC data." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.RED + f"Error processing {model_name} for {config}: {e}" + Style.RESET_ALL)

        # Check if any metrics were collected
        if not metrics:
            print(Fore.RED + "No AUC data available for comparison." + Style.RESET_ALL)
            return

        # Convert metrics to DataFrame for visualization
        df = pd.DataFrame(metrics)

        # Generate bar chart for Test AUC
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="Configuration", y="Test AUC", hue="Model", edgecolor="black")
        plt.title("Test AUC Comparison Across Configurations and Models")
        plt.ylabel("Test AUC")
        plt.xlabel("Configuration")
        plt.legend(title="Model")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    submenu()
