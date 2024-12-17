import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from graphviz import Source
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style
from modules.saving import ensure_dir, ASSETS_DIR

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
    data = data.copy()
    log_messages = []

    if 'first_purchase' in data.columns and 'last_purchase' in data.columns:
        data['customer_tenure_days'] = (data['last_purchase'] - data['first_purchase']).dt.days
        data['recency_days'] = (data['last_purchase'].max() - data['last_purchase']).dt.days
        data.drop(columns=['first_purchase', 'last_purchase'], inplace=True)
        log_messages.append("Converted 'first_purchase' and 'last_purchase' to 'customer_tenure_days' and 'recency_days'.")

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

    scaler = StandardScaler()
    for feature in numeric_features:
        if feature in data.columns:
            data[feature] = scaler.fit_transform(data[[feature]])
            log_messages.append(f"Scaled numeric feature '{feature}' using StandardScaler.")

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

    inf_count = data.isin([float('inf'), float('-inf')]).sum().sum()
    if inf_count > 0:
        data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        log_messages.append(f"Replaced {inf_count} infinite values in the dataset with NaN.")
    nan_count_before = data.isna().sum().sum()
    data = data.fillna(0)
    nan_count_after = data.isna().sum().sum()
    log_messages.append(f"Filled {nan_count_before - nan_count_after} remaining NaN values with 0.")

    global_logs["preprocessing"]["steps"] = log_messages

    print(Fore.CYAN + "\nPreprocessing Log:" + Style.RESET_ALL)
    for message in log_messages:
        print(Fore.GREEN + "✔ " + message + Style.RESET_ALL)

    return data

def clean_string_to_snake_case(input_string):
    """
    Convert a string to lowercase, replacing spaces and special characters with underscores.

    Args:
        input_string (str): The input string to clean.

    Returns:
        str: The cleaned string in snake_case format.
    """
    # Step 1: Convert to lowercase
    lowercased = input_string.lower()
    
    # Step 2: Replace all non-alphanumeric characters (except spaces) with a space
    no_specials = re.sub(r'[^a-z0-9\s]', ' ', lowercased)
    
    # Step 3: Replace spaces (including consecutive spaces) with a single underscore
    snake_case = re.sub(r'\s+', '_', no_specials)
    
    # Step 4: Trim leading/trailing underscores (if any)
    cleaned_string = snake_case.strip('_')
    
    return cleaned_string

def predict_customer_response(data, response='response'):
    print(Fore.CYAN + "\nPredicting Customer Response..." + Style.RESET_ALL)

    processed_data = None

    def main_submenu():
        while True:
            print(Fore.CYAN + "\nCustomer Response Prediction Submenu:" + Style.RESET_ALL)
            print("1. Preprocess Data")
            print("2. Model Selection Menu")
            print("6. Summarize Results of All Models")
            print("0. Return to Main Menu")

            choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)

            if choice == '1':
                nonlocal processed_data
                processed_data = preprocess_data_for_model(data)
                print(Fore.GREEN + "Data preprocessing completed. Ready for modeling." + Style.RESET_ALL)
            elif choice == '2':
                if processed_data is None:
                    print(Fore.RED + "Error: Data not preprocessed. Please select option 1 first." + Style.RESET_ALL)
                else:
                    model_selection_menu(processed_data, response)
            elif choice == '6':
                summarize_results()
            elif choice == '0':
                print(Fore.CYAN + "Returning to Main Menu..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Invalid choice! Please try again." + Style.RESET_ALL)

    def model_selection_menu(processed_data, response):
        while True:
            print(Fore.CYAN + "\nModel Selection Menu:" + Style.RESET_ALL)
            print("1. Logistic Regression (BLR)")
            print("2. Random Forest (RF)")
            print("3. Naive Bayes")
            print("4. Decision Tree")
            print("5. SVM")
            print("6. Decision Tree (Binned First)")
            print("7. Decision Tree (Binned After)")
            print("0. Return to Previous Menu")

            choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)

            if choice == '1':
                configuration_menu(processed_data, response, model_type='logistic_regression')
            elif choice == '2':
                configuration_menu(processed_data, response, model_type='random_forest')
            elif choice == '3':
                configuration_menu(processed_data, response, model_type='naive_bayes')
            elif choice == '4':
                configuration_menu(processed_data, response, model_type='decision_tree')
            elif choice == '5':
                configuration_menu(processed_data, response, model_type='svm')
            elif choice == '6':
                configuration_menu(processed_data, response, model_type='decision_tree_binned')
            elif choice == '7':
                configuration_menu(processed_data, response, model_type='decision_tree_binned_after')
            elif choice == '0':
                print(Fore.CYAN + "Returning to Previous Menu..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Invalid choice! Please try again." + Style.RESET_ALL)

    def configuration_menu(processed_data, response, model_type):
        while True:
            print(Fore.CYAN + f"\nConfiguration Menu for {model_type.capitalize()}:" + Style.RESET_ALL)
            print("1. Baseline")
            print("2. Recoded")
            print("3. Outlier Removed")
            print("4. Outlier Recoded Scaled")
            print("0. Return to Previous Menu")

            choice = input(Fore.YELLOW + "Enter your choice: " + Style.RESET_ALL)

            if choice in {'1', '2', '3', '4'}:
                config_map = {
                    '1': 'baseline',
                    '2': 'recoded',
                    '3': 'outlier_removed',
                    '4': 'outlier_recoded_scaled'
                }
                configuration_key = config_map[choice]
                run_selected_configuration(processed_data, response, model_type, configuration_key)
            elif choice == '0':
                print(Fore.CYAN + "Returning to Previous Menu..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Invalid choice! Please try again." + Style.RESET_ALL)

    def run_selected_configuration(processed_data, response, model_type, configuration_key):
        # Prompt the user for train/test ratio
        while True:
            try:
                train_size = float(input(Fore.YELLOW + "Enter train/test split ratio (e.g., 0.7 for 70% train): " + Style.RESET_ALL))
                if 0.0 < train_size < 1.0:
                    break
                else:
                    print(Fore.RED + "Invalid input. Please enter a value between 0.0 and 1.0." + Style.RESET_ALL)
            except ValueError:
                print(Fore.RED + "Invalid input. Please enter a numeric value." + Style.RESET_ALL)

        if configuration_key == 'baseline':
            # For model_type decision tree binned first we need to bin variables before performing other tasks
            # On the other hand, if we choose the model type with binned after, variables will be binned afted multic and significance
            if model_type == 'decision_tree_binned':
                processed_data = bin_numerical_variables(processed_data, response)

            # For baseline, we do significance checks and then call run_models
            filtered_data = check_and_handle_multicollinearity(processed_data, response)
            significant_vars = explore_feature_significance(filtered_data, response)

            # Bin variables after multicollinearity and significance checks
            if model_type == "decision_tree_binned_after":
                data = bin_numerical_variables(data, response)
        
            if not significant_vars:
                print(Fore.RED + "No significant variables found for baseline model." + Style.RESET_ALL)
                return
            
            # Here we select the model based on model_type (passed from previous menus)
            if model_type == 'logistic_regression':
                model = LogisticRegression(max_iter=1000)
                model_name = "Logistic Regression"
            elif model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_name = "Random Forest"
            elif model_type == 'naive_bayes':
                model = GaussianNB()
                model_name = "Naive Bayes"
            elif model_type == 'decision_tree':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree"
            elif model_type == 'svm':
                model = SVC(probability=True, random_state=42)
                model_name = "SVM"
            elif model_type == 'decision_tree_binned':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree (Binned First)"
            elif model_type == 'decision_tree_binned_after':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree (Binned After)"
            else:
                print(Fore.RED + "Unsupported model type for baseline configuration." + Style.RESET_ALL)
                return
            
            # Now call run_models with model and model_name
            run_models(filtered_data, significant_vars, response, configuration_key, model, model_name, train_size)

        elif configuration_key == 'recoded':
            # For model_type decision tree binned we need to bin variables before performing other tasks
            if model_type == 'decision_tree_binned':
                processed_data = bin_numerical_variables(processed_data, response)
            # Select model
            if model_type == 'logistic_regression':
                model = LogisticRegression(max_iter=1000)
                model_name = "Logistic Regression"
            elif model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_name = "Random Forest"
            elif model_type == 'naive_bayes':
                model = GaussianNB()
                model_name = "Naive Bayes"
            elif model_type == 'decision_tree':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree"
            elif model_type == 'svm':
                model = SVC(probability=True, random_state=42)
                model_name = "SVM"
            elif model_type == 'decision_tree_binned':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree (Binned First)"
            elif model_type == 'decision_tree_binned_after':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree (Binned After)"
            else:
                print(Fore.RED + "Unsupported model type for recoded configuration." + Style.RESET_ALL)
                return

            # Pass model, model_name, train_size to the specialized function
            run_models_with_recoded_variables(processed_data, response, model, model_name, train_size)

        elif configuration_key == 'outlier_removed':
            # For model_type decision tree binned we need to bin variables before performing other tasks
            if model_type == 'decision_tree_binned':
                processed_data = bin_numerical_variables(processed_data, response)
            # Select model
            if model_type == 'logistic_regression':
                model = LogisticRegression(max_iter=1000)
                model_name = "Logistic Regression"
            elif model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_name = "Random Forest"
            elif model_type == 'naive_bayes':
                model = GaussianNB()
                model_name = "Naive Bayes"
            elif model_type == 'decision_tree':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree"
            elif model_type == 'svm':
                model = SVC(probability=True, random_state=42)
                model_name = "SVM"
            elif model_type == 'decision_tree_binned':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree (Binned First)"
            elif model_type == 'decision_tree_binned_after':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree (Binned After)"
            else:
                print(Fore.RED + "Unsupported model type for outlier_removed configuration." + Style.RESET_ALL)
                return

            # Pass model, model_name, train_size to the specialized function
            run_models_with_outlier_removal(processed_data, response, model, model_name, train_size)

        elif configuration_key == 'outlier_recoded_scaled':
            # For model_type decision tree binned we need to bin variables before performing other tasks
            if model_type == 'decision_tree_binned':
                processed_data = bin_numerical_variables(processed_data, response)
            # Select model based on model_type
            if model_type == 'logistic_regression':
                model = LogisticRegression(max_iter=1000)
                model_name = "Logistic Regression"
            elif model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_name = "Random Forest"
            elif model_type == 'naive_bayes':
                model = GaussianNB()
                model_name = "Naive Bayes"
            elif model_type == 'decision_tree':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree"
            elif model_type == 'svm':
                model = SVC(probability=True, random_state=42)
                model_name = "SVM"
            elif model_type == 'decision_tree_binned':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree (Binned First)"
            elif model_type == 'decision_tree_binned_after':
                model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree (Binned After)"
            else:
                print(Fore.RED + "Unsupported model type for outlier_recoded_scaled configuration." + Style.RESET_ALL)
                return

            # Now call no_outliers_and_recoded_scaled_variables with model, model_name, and train_size
            no_outliers_and_recoded_scaled_variables(processed_data, response, model, model_name, train_size)

    def check_and_handle_multicollinearity(data, response):
        print(Fore.CYAN + "\nPerforming Multicollinearity Check..." + Style.RESET_ALL)
        log_messages = []
        X = data.select_dtypes(include=[np.number]).drop(columns=[response], errors='ignore')

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

        final_vif = vif_data[-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x="VIF", y="Feature", data=final_vif.sort_values("VIF", ascending=False), palette="coolwarm")
        plt.axvline(x=5, color='red', linestyle='--', label='VIF Threshold (5)')
        plt.title("Final Variance Inflation Factor (VIF) After Multicollinearity Check")
        plt.xlabel("VIF Value")
        plt.ylabel("Feature")
        plt.legend()
        plt.grid(alpha=0.3)
        vif_dir = os.path.join(ASSETS_DIR, "vif_plots")
        ensure_dir(vif_dir)
        plt.savefig(os.path.join(vif_dir, "final_vif_plot.png"))
        plt.show()

        print(Fore.CYAN + "\nMulticollinearity Log:" + Style.RESET_ALL)
        for log in log_messages:
            print(Fore.GREEN + "✔ " + log + Style.RESET_ALL)

        filtered_data = pd.concat([X, data[response]], axis=1)
        return filtered_data

    def explore_feature_significance(data, response):
        print(Fore.CYAN + "\nExploring Feature Significance..." + Style.RESET_ALL)
        X = data.drop(columns=[response]) if response in data.columns else data
        y = data[response]

        X_with_const = sm.add_constant(X)
        logit_model = sm.Logit(y, X_with_const).fit(disp=False)

        pvalues = logit_model.pvalues[1:]
        pvalues.sort_values(inplace=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=-np.log10(pvalues.values), y=pvalues.index, palette="coolwarm")
        plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='-log10(0.05)')
        plt.title("Feature Significance (-log10(p-values))")
        plt.xlabel("-log10(p-value)")
        plt.ylabel("Variable")
        plt.legend()
        plt.grid(alpha=0.3)
        significance_dir = os.path.join(ASSETS_DIR, "feature_significance")
        ensure_dir(significance_dir)
        plt.savefig(os.path.join(significance_dir, "feature_significance_barplot.png"))
        plt.show()

        significant_vars = [
            var for var in logit_model.pvalues.index
            if var != 'const' and logit_model.pvalues[var] < 0.05
        ]
        global_logs["significance"]["variables"] = significant_vars

        print(Fore.CYAN + "\nLogistic Regression Summary:" + Style.RESET_ALL)
        print(logit_model.summary())
        return significant_vars

    def run_models(data, significant_vars, response, configuration_key, model, model_name, train_size):
        print(Fore.CYAN + "\nValidating Significant Variables Against Data Columns..." + Style.RESET_ALL)
        missing_vars = [var for var in significant_vars if var not in data.columns]
        if missing_vars:
            print(Fore.RED + f"Error: The following significant variables are missing from the dataset: {missing_vars}" + Style.RESET_ALL)
            print(Fore.CYAN + "Available Columns in Data:" + Style.RESET_ALL)
            print(Fore.GREEN + ", ".join(data.columns) + Style.RESET_ALL)
            return

        significant_vars = [var for var in significant_vars if var in data.columns]
        X = data[significant_vars]
        y = data[response]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_size), random_state=42)
        model.fit(X_train, y_train)
        
        # **Decision Tree Visualization**
        if isinstance(model, DecisionTreeClassifier):
            print(Fore.CYAN + "\nVisualizing Decision Tree..." + Style.RESET_ALL)
            visualize_decision_tree(
                model=model,
                feature_names=X.columns,
                class_names=["No", "Yes"],
                config_key=configuration_key,
                file_name=f"{clean_string_to_snake_case(model_name)}_{configuration_key}"
            )
        
        train_metrics = evaluate_model(y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1], model_name, dataset_type="Train")
        test_metrics = evaluate_model(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1], model_name, train_metrics=train_metrics, dataset_type="Test")

        if global_logs["models"][configuration_key] is None:
            global_logs["models"][configuration_key] = {}

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
        print(Fore.CYAN + f"\n{model_name} Evaluation on {dataset_type} Data:" + Style.RESET_ALL)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[optimal_idx]

        print(f"Accuracy: {accuracy:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, Balanced Accuracy: {balanced_accuracy:.2f}")
        print(f"Best Threshold: {best_threshold:.2f}")

        if train_metrics is not None and dataset_type == "Test":
            print(Fore.YELLOW + "\nOverfitting Assessment:" + Style.RESET_ALL)
            print(f"Train-Test Accuracy Gap: {train_metrics['accuracy'] - accuracy:.2f}")
            print(f"Train-Test Balanced Accuracy Gap: {train_metrics['balanced_accuracy'] - balanced_accuracy:.2f}")
            if abs(train_metrics['accuracy'] - accuracy) > 0.1:
                print(Fore.RED + "Warning: Potential Overfitting Detected (Accuracy Gap > 0.1)." + Style.RESET_ALL)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f"Confusion Matrix ({model_name}, {dataset_type})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_dir = os.path.join(ASSETS_DIR, "confusion_matrices")
        ensure_dir(cm_dir)
        plt.savefig(os.path.join(cm_dir, f"confusion_matrix_{clean_string_to_snake_case(model_name)}_{dataset_type.lower()}.png"))
        plt.show()

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f"ROC Curve ({model_name}, {dataset_type})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        roc_dir = os.path.join(ASSETS_DIR, "roc_curves")
        ensure_dir(roc_dir)
        plt.savefig(os.path.join(roc_dir, f"roc_curve_{clean_string_to_snake_case(model_name)}_{dataset_type.lower()}.png"))
        plt.show()

        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "best_threshold": best_threshold,
        }

    def run_models_with_recoded_variables(data, response, model, model_name, train_size):
        data = preprocess_data_for_model(data)

        if 'num_orders' in data.columns and 'recency_days' in data.columns:
            data['recency_ratio'] = data['recency_days'] / (data['num_orders'] + 1e-5)
            data['complaint_ratio'] = data['n_comp'] / (data['num_orders'] + 1e-5)
            print(Fore.GREEN + "✔ Added 'recency_ratio' and 'complaint_ratio' to the dataset." + Style.RESET_ALL)

        scaler = StandardScaler()
        for feature in ['recency_ratio', 'complaint_ratio']:
            if feature in data.columns:
                data[feature] = scaler.fit_transform(data[[feature]])
                print(Fore.GREEN + f"✔ Scaled feature '{feature}' using StandardScaler." + Style.RESET_ALL)

        data = check_and_handle_multicollinearity(data, response)
        significant_vars = explore_feature_significance(data, response)

        # Bin variables after multicollinearity and significance checks
        if model_name == "Decision Tree (Binned After)":
            data = bin_numerical_variables(data, response)
        
        if not significant_vars:
            print(Fore.RED + "No significant variables found after multicollinearity checks." + Style.RESET_ALL)
            return

        configuration_key = "recoded"
        # Now include model, model_name, train_size:
        run_models(data, significant_vars, response, configuration_key, model, model_name, train_size)

    def run_models_with_outlier_removal(data, response, model, model_name, train_size):
        data = preprocess_data_for_model(data)

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

        data = check_and_handle_multicollinearity(data, response)
        significant_vars = explore_feature_significance(data, response)

        # Bin variables after multicollinearity and significance checks
        if model_name == "Decision Tree (Binned After)":
            data = bin_numerical_variables(data, response)

        if not significant_vars:
            print(Fore.RED + "No significant variables found after multicollinearity checks." + Style.RESET_ALL)
            return

        configuration_key = "outlier_removed"
        run_models(data, significant_vars, response, configuration_key, model, model_name, train_size)

    def no_outliers_and_recoded_scaled_variables(data, response, model, model_name, train_size):
        processed_data = preprocess_data_for_model(data)

        if 'num_orders' in processed_data.columns and 'recency_days' in processed_data.columns:
            processed_data['recency_ratio'] = processed_data['recency_days'] / (processed_data['num_orders'] + 1e-5)
            processed_data['complaint_ratio'] = processed_data['n_comp'] / (processed_data['num_orders'] + 1e-5)
            print(Fore.GREEN + "✔ Added 'recency_ratio' and 'complaint_ratio' to the dataset." + Style.RESET_ALL)

        scaler = StandardScaler()
        for feature in ['recency_ratio', 'complaint_ratio']:
            if feature in processed_data.columns:
                processed_data[feature] = scaler.fit_transform(processed_data[[feature]])
                print(Fore.GREEN + f"✔ Scaled numeric feature '{feature}' using StandardScaler." + Style.RESET_ALL)

        numerical_fields = processed_data.select_dtypes(include=['float64', 'int64']).columns
        outlier_logs = []

        for field in numerical_fields:
            if field == response:
                continue
            Q1 = processed_data[field].quantile(0.25)
            Q3 = processed_data[field].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((processed_data[field] < lower_bound) | (processed_data[field] > upper_bound))
            outlier_count = outliers.sum()
            if outlier_count > 0:
                processed_data = processed_data[~outliers]
                outlier_logs.append(f"Removed {outlier_count} outliers from '{field}'. Remaining rows: {len(processed_data)}")

        if outlier_logs:
            print(Fore.GREEN + "\nOutlier Removal Summary:" + Style.RESET_ALL)
            for log in outlier_logs:
                print(Fore.GREEN + "✔ " + log + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "No outliers detected for numerical fields." + Style.RESET_ALL)

        print(Fore.CYAN + "\nChecking Multicollinearity..." + Style.RESET_ALL)
        filtered_data = check_and_handle_multicollinearity(processed_data, response)

        print(Fore.CYAN + "\nPerforming Significance Testing..." + Style.RESET_ALL)
        significant_vars = explore_feature_significance(filtered_data, response)

        # Bin variables after multicollinearity and significance checks
        if model_name == "Decision Tree (Binned After)":
            filtered_data = bin_numerical_variables(data, response)
        
        if significant_vars:
            print(Fore.CYAN + "\nRunning Models on Cleaned Dataset with Recoded Variables..." + Style.RESET_ALL)
            # Now we have model, model_name, and train_size from run_selected_configuration()
            run_models(filtered_data, significant_vars, response, "outlier_recoded_scaled", model, model_name, train_size)
        else:
            print(Fore.RED + "No significant variables identified after outlier removal and multicollinearity checks." + Style.RESET_ALL)

    def summarize_results():
        print(Fore.CYAN + "\nSummary of All Model Runs:" + Style.RESET_ALL)
        print(Fore.CYAN + "\nPreprocessing Summary:" + Style.RESET_ALL)
        preprocessing_steps = global_logs.get("preprocessing", {}).get("steps", [])
        if preprocessing_steps:
            for step in preprocessing_steps:
                print(Fore.GREEN + f"✔ {step}" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "No preprocessing steps logged." + Style.RESET_ALL)

        print(Fore.CYAN + "\nMulticollinearity Checks:" + Style.RESET_ALL)
        vif_results = global_logs.get("multicollinearity", {}).get("vif")
        if vif_results:
            print(Fore.GREEN + "✔ Final VIF Values:" + Style.RESET_ALL)
            for record in vif_results:
                print(Fore.GREEN + f"   {record['Feature']}: VIF = {record['VIF']:.2f}" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "No multicollinearity results logged." + Style.RESET_ALL)

        print(Fore.CYAN + "\nModel Performance Comparison:" + Style.RESET_ALL)
        model_results = global_logs.get("models", {})
        all_metrics = []

        for config, models in model_results.items():
            if models:
                for model_name, metrics in models.items():
                    try:
                        print(Fore.GREEN + f"\n{config.capitalize()} - {model_name.replace('_', ' ').capitalize()} Model Results:" + Style.RESET_ALL)
                        print(f"Train Accuracy: {metrics['train_accuracy']:.2f}")
                        print(f"Test Accuracy: {metrics['test_accuracy']:.2f}")
                        print(f"Overfitting Gap: {metrics['overfitting_gap']:.2f}")
                        print(f"Balanced Accuracy Gap: {metrics['balanced_accuracy_gap']:.2f}")
                        print(f"Variables Used: {', '.join(metrics['variables_used'])}")

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

        if all_metrics:
            generate_aggregated_performance_chart(all_metrics)
        else:
            print(Fore.RED + "No performance metrics available for visualization." + Style.RESET_ALL)

    def generate_aggregated_performance_chart(metrics):
        df = pd.DataFrame(metrics)

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
        aggregated_dir = os.path.join(ASSETS_DIR, "aggregated_metrics")
        ensure_dir(aggregated_dir)
        plt.savefig(os.path.join(aggregated_dir, "test_accuracy_comparison.png"))
        plt.show()

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
        aggregated_dir = os.path.join(ASSETS_DIR, "aggregated_metrics")
        ensure_dir(aggregated_dir)
        plt.savefig(os.path.join(aggregated_dir, "overfitting_gap_comparison.png"))
        plt.show()

    def preprocess_for_configuration(data, config_key, response="response"):
        processed_data = preprocess_data_for_model(data)

        if config_key == "baseline":
            processed_data = check_and_handle_multicollinearity(processed_data, response)
            significant_vars = explore_feature_significance(processed_data, response)
            if significant_vars:
                processed_data = processed_data[significant_vars + [response]]
            return processed_data

        if config_key == "recoded":
            if 'num_orders' in processed_data.columns and 'recency_days' in processed_data.columns:
                processed_data['recency_ratio'] = processed_data['recency_days'] / (processed_data['num_orders'] + 1e-5)
                processed_data['complaint_ratio'] = processed_data['n_comp'] / (processed_data['num_orders'] + 1e-5)

            numerical_fields = ['recency_ratio', 'complaint_ratio']
            scaler = StandardScaler()
            for field in numerical_fields:
                if field in processed_data.columns:
                    processed_data[field] = scaler.fit_transform(processed_data[[field]])

            processed_data = check_and_handle_multicollinearity(processed_data, response)
            significant_vars = explore_feature_significance(processed_data, response)
            if significant_vars:
                processed_data = processed_data[significant_vars + [response]]
            return processed_data

        if config_key == "outlier_removed":
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

            processed_data = check_and_handle_multicollinearity(processed_data, response)
            significant_vars = explore_feature_significance(processed_data, response)
            if significant_vars:
                processed_data = processed_data[significant_vars + [response]]
            return processed_data

        if config_key == "outlier_recoded_scaled":
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

    def visualize_decision_tree(model, feature_names, class_names, config_key=None, file_name="tree_visualization"):

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
        # Decide on the output directory for decision trees
        dt_dir = os.path.join(ASSETS_DIR, "decision_trees")
        ensure_dir(dt_dir)
        output_path = os.path.join(dt_dir, file_name)
        graph.render(output_path, format="png", cleanup=True)
        print(f"Decision tree visualization saved as {file_name}.png")

    def bin_numerical_variables(data, response="response"):
        print(Fore.CYAN + "\nBinning Numerical Variables..." + Style.RESET_ALL)
        binned_data = data.copy()
        numerical_cols = binned_data.select_dtypes(include=['float64', 'int64']).columns
        binned_logs = []

        for col in numerical_cols:
            if col == response:
                continue

            Q1 = binned_data[col].quantile(0.25)
            Q3 = binned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            non_outliers = binned_data[(binned_data[col] >= lower_bound) & (binned_data[col] <= upper_bound)][col]

            bins = [-float('inf'), non_outliers.quantile(0.25), non_outliers.median(), non_outliers.quantile(0.75), float('inf')]
            labels = [f"{col}_low", f"{col}_medium", f"{col}_high", f"{col}_very_high"]

            try:
                binned_data[col] = pd.cut(
                    binned_data[col],
                    bins=bins,
                    labels=labels,
                    include_lowest=True,
                )
                le = LabelEncoder()
                binned_data[col] = le.fit_transform(binned_data[col].astype(str))
                binned_logs.append(f"Binned and encoded '{col}' into quartile-based categories: {labels}")
            except ValueError as e:
                binned_logs.append(f"Skipped binning for '{col}' due to duplicate or invalid bin edges: {e}")

        print(Fore.GREEN + "\n✔ Binning Summary:" + Style.RESET_ALL)
        for log in binned_logs:
            print(Fore.GREEN + "✔ " + log + Style.RESET_ALL)

        return binned_data

    main_submenu()
