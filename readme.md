
# **Customer Campaign Analysis Algorithm**

## **Overview**
This algorithm facilitates the analysis of customer behavior and campaign responses using datasets of retail transactions and campaign performance. It performs data preprocessing, feature engineering, exploratory data analysis, and prediction modeling. The algorithm is menu-driven, offering interactive options for performing various tasks.

---

## **Features**
1. **Load and Preprocess Data**
2. **Convert Data Types**
3. **Handle Missing Values and Invalid Data**
4. **Perform Integrity Checks and Handle Duplicates**
5. **Perform JOIN Between Datasets**
6. **Perform Exploratory Data Analysis (EDA)**
7. **Predict Customer Response**
8. **Run Full Pipeline**

---

## **Menu Options**

### **1. Load and Preprocess Data**
- **Purpose**: Load datasets and display their initial dimensions.
- **Steps**:
  1. Load the **Online Retail Data** and **Campaign Response Data** from CSV files.
  2. Print the number of rows and columns in each dataset.

---

### **2. Convert Data Types**
- **Purpose**: Ensure data types are correct and allow interactive recoding for numerical variables.
- **Steps**:
  1. Convert key columns (e.g., `CustomerID`, `Country`, `InvoiceDate`) to appropriate types.
  2. For numerical variables, provide user options:
     - **Keep as numerical**
     - **Convert to categorical** (e.g., low, medium, high)
     - **Convert to boolean** (e.g., based on a threshold)

---

### **3. Handle Missing Values and Invalid Data**
- **Purpose**: Identify and handle missing or invalid data.
- **Steps**:
  1. Identify missing values for all columns.
  2. For identifier columns (e.g., `CustomerID`), options:
     - **Fill missing values with random unique identifiers**.
     - **Remove rows with missing identifiers**.
  3. For other columns:
     - **Remove rows with missing values**.
     - **Fill missing values with a default value** (based on data type).
     - **Do nothing**.

---

### **4. Perform Integrity Checks and Handle Duplicates**
- **Purpose**: Ensure data integrity by handling duplicates and outliers.
- **Steps**:
  1. Report and remove duplicate rows.
  2. Remove rows with negative values in `UnitPrice`.

---

### **5. Perform JOIN Between Datasets**
- **Purpose**: Merge customer features with campaign response data.
- **Steps**:
  1. **Derive Customer Features**:
     - **Total Sales**: Total spending by each customer.
     - **Unique Products**: Number of unique products purchased.
     - **Invoices**: Number of unique invoices generated.
     - Additional features like customer tenure, recency, and return rates.
  2. Perform an **inner join** with campaign response data using `CustomerID`.
  3. Display the dimensions and data types of the resulting dataset.
  4. **Calculate Campaign Response Rate**:
     - Display the percentage of customers who responded positively to the campaign.

---

### **6. Perform Exploratory Data Analysis (EDA)**
- **Purpose**: Provide insights into the data through visualizations and statistical checks.
- **Steps**:
  1. **Correlation Heatmap**: Display correlations between variables.
  2. **QQ Plots**: Assess normality for numerical variables with or without outliers.
  3. **Boxplots by Response**: Visualize distributions of variables grouped by campaign response.

---

### **7. Predict Customer Response**
- **Purpose**: Build predictive models to classify customer responses.
- **Steps**:
  1. **Preprocess Data**: Prepare data by scaling features and handling missing values.
  2. **Explore Feature Significance**:
     - Calculate p-values for features using logistic regression.
     - Visualize significant and non-significant variables.
  3. **Check Multicollinearity**:
     - Calculate Variance Inflation Factor (VIF) for features.
     - Optionally remove features with high multicollinearity.
  4. **Train Models**:
     - **Logistic Regression**: Evaluate accuracy, precision, recall, F1 score, and ROC-AUC.
     - **Random Forest**: Evaluate performance and feature importance.

---

### **8. Run Full Pipeline**
- **Purpose**: Execute all steps sequentially.
- **Steps**:
  1. Load and preprocess data.
  2. Convert data types and handle missing values.
  3. Perform integrity checks.
  4. Derive customer features and join datasets.
  5. Conduct EDA.
  6. Build and evaluate predictive models.

---

## **Prerequisites**
- Python 3.8 or later
- Required Python libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - statsmodels
  - colorama
  - rich

---

## **How to Use**
1. Run `main.py`.
2. Follow the menu options to perform tasks.
3. Use **Option 8** to execute the full pipeline.

---

## **Dataset Information**
1. **Online Retail Data**:
   - Variables: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.
2. **Campaign Response Data**:
   - Variables: `CustomerID`, `response`, `loyalty`, `nps`.

---

## **Sample Outputs**
- **Data Loading**:
  ```
  Initial dimensions of Online Retail Data: 50,000 rows, 8 columns.
  Initial dimensions of Campaign Response Data: 10,000 rows, 4 columns.
  ```
- **Campaign Response Rate**:
  ```
  Overall Campaign Response Rate: 25.50%
  Response = 1 for 2,550 out of 10,000 customers.
  ```
- **Feature Engineering**:
  ```
  Customer Features Overview:
  CustomerID | Total Spending | Number of Orders | Recency | Return Rate | ...
  ```
