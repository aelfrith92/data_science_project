import sys
import os
import pandas as pd
from colorama import Fore, Style, init

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.integrity_checks import perform_integrity_checks
from modules.preprocessing import convert_data_types, convert_campaign_data_types, print_data_overview
from modules.missing_data_handling import process_missing_data
from modules.join_operations import derive_customer_features, perform_customer_campaign_join
from modules.predict_campaign_response import predict_customer_response
from modules.exploratory_analysis import analyze_data

# In the main menu:
def main():
    while True:
        print_menu()
        choice = input(Fore.CYAN + "Enter your choice (0-8): " + Style.RESET_ALL).strip()

        if not choice.isdigit() or int(choice) not in range(0, 9):
            print(Fore.RED + "Invalid choice, please select a valid option (0-8)." + Style.RESET_ALL)
            continue

        choice = int(choice)

        if choice == 1:
            online_retail_data, campaign_response_data = load_data()
            print_initial_dimensions(online_retail_data, "Online Retail")
            print_initial_dimensions(campaign_response_data, "Campaign Response")

        elif choice == 2:
            if 'online_retail_data' in locals() and 'campaign_response_data' in locals():
                online_retail_data = convert_data_types(online_retail_data)
                campaign_response_data = convert_campaign_data_types(campaign_response_data)
                print_data_overview(online_retail_data, "Online Retail")
                print_data_overview(campaign_response_data, "Campaign Response")
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 1 first." + Style.RESET_ALL)

        elif choice == 3:
            if 'online_retail_data' in locals() and 'campaign_response_data' in locals():
                online_retail_data = process_missing_data(online_retail_data)
                campaign_response_data = process_missing_data(campaign_response_data)
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 1 first." + Style.RESET_ALL)

        elif choice == 4:
            if 'online_retail_data' in locals() and 'campaign_response_data' in locals():
                online_retail_data, campaign_response_data = perform_integrity_checks(online_retail_data, campaign_response_data)
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 1 first." + Style.RESET_ALL)

        elif choice == 5:
            if 'online_retail_data' in locals() and 'campaign_response_data' in locals():
                customer_features = derive_customer_features(online_retail_data)

                if customer_features is not None:
                    joined_data = perform_customer_campaign_join(customer_features, campaign_response_data)
                else:
                    print(Fore.RED + "Error deriving customer features. Please check the data integrity." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 1 first." + Style.RESET_ALL)

        elif choice == 6:  # Exploratory Data Analysis (EDA)
            if 'joined_data' in locals():
                analyze_data(joined_data, response='response')
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 5 to perform the JOIN first." + Style.RESET_ALL)

        elif choice == 7:  # Predict Customer Response (ML)
            if 'joined_data' in locals():
                predict_customer_response(joined_data, response='response')
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 5 to perform the JOIN first." + Style.RESET_ALL)

        elif choice == 8:  # Full Pipeline (Run All)
            online_retail_data, campaign_response_data = load_data()
            online_retail_data = convert_data_types(online_retail_data)
            campaign_response_data = convert_campaign_data_types(campaign_response_data)
            online_retail_data = process_missing_data(online_retail_data)
            campaign_response_data = process_missing_data(campaign_response_data)
            online_retail_data, campaign_response_data = perform_integrity_checks(online_retail_data, campaign_response_data)
            customer_features = derive_customer_features(online_retail_data)

            if customer_features is not None:
                joined_data = perform_customer_campaign_join(customer_features, campaign_response_data)

                if joined_data is not None:
                    analyze_data(joined_data, response='response')
                    predict_customer_response(joined_data, response='response')
                else:
                    print(Fore.RED + "Error joining customer features with campaign response data." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Error deriving customer features. Please check the data integrity." + Style.RESET_ALL)

        elif choice == 0:
            print(Fore.CYAN + "\nExiting the program. Goodbye!" + Style.RESET_ALL)
            break

# Initialize colorama
init(autoreset=True)

def print_section_header(header_text):
    print(f"\n{Fore.CYAN}{'='*len(header_text)}")
    print(header_text)
    print(f"{'='*len(header_text)}\n")

def print_menu():
    print(Fore.YELLOW + "\nData Processing Menu:")
    print(f"{Fore.CYAN}1.{Style.RESET_ALL} Load and Preprocess Data")
    print(f"{Fore.CYAN}2.{Style.RESET_ALL} Convert Data Types")
    print(f"{Fore.CYAN}3.{Style.RESET_ALL} Handle Missing Values and Invalid Data")
    print(f"{Fore.CYAN}4.{Style.RESET_ALL} Perform Integrity Checks and Handle Duplicates")
    print(f"{Fore.CYAN}5.{Style.RESET_ALL} Perform JOIN Between Datasets")
    print(f"{Fore.CYAN}6.{Style.RESET_ALL} Perform Exploratory Data Analysis (EDA)")
    print(f"{Fore.CYAN}7.{Style.RESET_ALL} Predict Customer Response")
    print(f"{Fore.CYAN}8.{Style.RESET_ALL} Full Pipeline (Run All)")
    print(f"{Fore.CYAN}0.{Style.RESET_ALL} Exit")

def load_data():
    print_section_header("Data Loading and Initial Preprocessing")
    online_retail_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'online_retail.csv'))
    campaign_response_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'campaign_response.csv'))
    print(Fore.GREEN + "Data has been successfully loaded and initial preprocessing is done." + Style.RESET_ALL)
    return online_retail_data, campaign_response_data

def format_number(num):
    return f"{num:,}"

def print_initial_dimensions(df, dataset_name):
    print(f"\n{Fore.CYAN}Initial dimensions of {dataset_name} Data: {format_number(df.shape[0])} rows, {format_number(df.shape[1])} columns" + Style.RESET_ALL)

def main():

    while True:
        print_menu()
        choice = input(Fore.CYAN + "Enter your choice (0-8): " + Style.RESET_ALL)

        if choice == '1':
            online_retail_data, campaign_response_data = load_data()
            print_initial_dimensions(online_retail_data, "Online Retail")
            print_initial_dimensions(campaign_response_data, "Campaign Response")

        elif choice == '2':
            if 'online_retail_data' in locals() and 'campaign_response_data' in locals():
                online_retail_data = convert_data_types(online_retail_data)
                campaign_response_data = convert_campaign_data_types(campaign_response_data)
                print_data_overview(online_retail_data, "Online Retail")
                print_data_overview(campaign_response_data, "Campaign Response")
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 1 first." + Style.RESET_ALL)

        elif choice == '3':
            if 'online_retail_data' in locals() and 'campaign_response_data' in locals():
                online_retail_data = process_missing_data(online_retail_data)
                campaign_response_data = process_missing_data(campaign_response_data)
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 1 first." + Style.RESET_ALL)

        elif choice == '4':
            if 'online_retail_data' in locals() and 'campaign_response_data' in locals():
                online_retail_data, campaign_response_data = perform_integrity_checks(online_retail_data, campaign_response_data)
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 1 first." + Style.RESET_ALL)

        elif choice == '5':
            if 'online_retail_data' in locals() and 'campaign_response_data' in locals():
                customer_features = derive_customer_features(online_retail_data)

                if customer_features is not None:
                    joined_data = perform_customer_campaign_join(customer_features, campaign_response_data)
                else:
                    print(Fore.RED + "Error deriving customer features. Please check the data integrity." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 1 first." + Style.RESET_ALL)

        elif choice == '6':  # Exploratory Data Analysis (EDA)
            if 'joined_data' in locals():
                analyze_data(joined_data, response='response')  # Call the new visualization function
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 5 to perform the JOIN first." + Style.RESET_ALL)

        elif choice == '7':  # Predict Customer Response (ML)
            if 'joined_data' in locals():
                predict_customer_response(joined_data, response='response')  # Updated function
            else:
                print(Fore.RED + "Data is not loaded. Please choose option 5 to perform the JOIN first." + Style.RESET_ALL)

        elif choice == '8':  # Full Pipeline (Run All)
            online_retail_data, campaign_response_data = load_data()
            online_retail_data = convert_data_types(online_retail_data)
            campaign_response_data = convert_campaign_data_types(campaign_response_data)
            online_retail_data = process_missing_data(online_retail_data)
            campaign_response_data = process_missing_data(campaign_response_data)
            online_retail_data, campaign_response_data = perform_integrity_checks(online_retail_data, campaign_response_data)
            customer_features = derive_customer_features(online_retail_data)

            if customer_features is not None:
                joined_data = perform_customer_campaign_join(customer_features, campaign_response_data)

                if joined_data is not None:
                    analyze_data(joined_data, response='response')  # UPDATED FUNCTION NAME
                    predict_customer_response(joined_data, response='response')
                else:
                    print(Fore.RED + "Error joining customer features with campaign response data." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Error deriving customer features. Please check the data integrity." + Style.RESET_ALL)

        elif choice == '0':
            print(Fore.CYAN + "\nExiting the program. Goodbye!" + Style.RESET_ALL)
            break

        else:
            print(Fore.RED + "Invalid choice, please select a valid option." + Style.RESET_ALL)

if __name__ == "__main__":
    main()