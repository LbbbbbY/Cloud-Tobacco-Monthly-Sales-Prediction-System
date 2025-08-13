# Cloud Tobacco Monthly Sales Prediction System

## Project Overview

This project is a machine learning-based monthly sales prediction system for cloud tobacco products, utilizing XGBoost algorithm to model and predict tobacco sales data across multiple cities. The system processes historical sales data and provides scientific prediction support for tobacco sales management through feature engineering and regression analysis.

## Project Structure

```
├── 代码及基础数据/ (Code and Data)
│   ├── dataDeal/
│   │   └── __init__.py          # Data processing module
│   ├── multC_multM_pre.py       # XGBoost sales prediction main program
│   ├── multC_multM_fit.py       # Prediction vs policy value regression analysis
│   ├── 月/ (Monthly Data)
│   │   ├── 云烟（软珍品）.csv    # Cloud Tobacco (Soft Premium) sales data
│   │   ├── 软大重九.csv         # Soft Da Zhong Jiu sales data
│   │   ├── 软玉.csv             # Soft Yu sales data
│   │   └── 全国地市名.csv       # City name mapping table
│   └── 说明文档.txt             # Project documentation
├── 结果数据/ (Results)           # Prediction results output directory
│   ├── 云烟（软珍品）-所有地市-fit结果-202312-202408.xlsx
│   ├── 云烟（软珍品）-所有地市-xgboost销量预测结果-V3-202311-202408.xlsx
│   ├── 软大重九-所有地市-fit结果-202312-202408.xlsx
│   ├── 软大重九-所有地市-xgboost销量预测结果-V3-202311-202408.xlsx
│   ├── 软玉-所有地市-fit结果-202312-202408.xlsx
│   └── 软玉-所有地市-xgboost销量预测结果-V3-202311-202408.xlsx
└── 云产烟月度销售预测.docx      # Project documentation
```

## Features

### 1. Multi-dimensional Data Prediction
- **Multi-brand Prediction**: Supports multiple tobacco brands including Cloud Tobacco (Soft Premium), Soft Da Zhong Jiu, and Soft Yu
- **Multi-city Prediction**: Covers sales data from multiple cities across China
- **Multi-temporal Analysis**: Supports monthly and annual prediction analysis

### 2. Intelligent Feature Engineering
- **Historical Data Features**: Utilizes historical sales data for feature construction
- **Time Series Features**: Includes month-over-month, year-over-year, and moving average features
- **Statistical Features**: Standard deviation, slope, and other statistical indicators
- **Business Features**: Order fulfillment rate, monthly ratio, cumulative ratio, and other business metrics

### 3. Machine Learning Algorithms
- **XGBoost Algorithm**: Uses gradient boosting decision trees for sales prediction
- **Linear Regression**: For regression analysis between predicted and policy values
- **Polynomial Features**: Supports polynomial feature transformation

### 4. Prediction Accuracy Assessment
- **Standard Error Calculation**: Evaluates prediction accuracy
- **R² Metrics**: Measures model fitting goodness
- **Accuracy Statistics**: Calculates prediction accuracy distribution

## Data Format

### Input Data Format
CSV files contain the following fields:
- `pcom_id`: Province ID
- `com_id`: City ID
- `month`: Month (format: YYYYMM)
- `qty_ord_item`: Order quantity
- `need_ord_item`: Demand quantity

### Output Data Format
Excel files contain prediction results, including:
- City information (province, city name)
- Prediction month
- Predicted value
- Actual value
- Accuracy rate

## Technology Stack

### Core Dependencies
- **Python 3.7+**
- **NumPy**: Numerical computing
- **Pandas**: Data processing
- **XGBoost**: Machine learning algorithm
- **Scikit-learn**: Machine learning toolkit
- **Matplotlib**: Data visualization

### Custom Modules
- **dataDeal**: Data processing toolkit, including data cleaning and feature calculation functions

## Usage

### 1. Environment Setup

#### Method 1: Using Conda (Recommended)
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate tobacco-prediction
```

#### Method 2: Using pip
```bash
# Install dependencies
pip install -r requirements.txt
```

#### Method 3: Using Docker
```bash
# Build image
docker build -t tobacco-prediction .

# Run container
docker run -v $(pwd)/代码及基础数据:/app/代码及基础数据 -v $(pwd)/结果数据:/app/结果数据 tobacco-prediction
```

#### Method 4: Using Docker Compose
```bash
# Run prediction service
docker-compose up tobacco-prediction

# Run regression analysis service
docker-compose up tobacco-regression
```

### 2. Data Preparation
Place sales data CSV files in the `代码及基础数据/月/` directory

### 3. Run Prediction
```bash
# Sales prediction
python 代码及基础数据/multC_multM_pre.py

# Regression analysis
python 代码及基础数据/multC_multM_fit.py
```

### 4. View Results
Prediction results will be saved in the `结果数据/` directory, including:
- XGBoost prediction results
- Fitting analysis results
- Accuracy statistics charts

## Prediction Workflow

1. **Data Preprocessing**
   - Data cleaning and standardization
   - Calculate order fulfillment rate
   - Generate time series features

2. **Feature Engineering**
   - Build historical data features
   - Calculate month-over-month and year-over-year indicators
   - Generate statistical features

3. **Model Training**
   - Train model using XGBoost algorithm
   - Parameter tuning and validation

4. **Prediction Output**
   - Generate prediction results
   - Calculate prediction accuracy
   - Output analysis reports

## Project Highlights

- **High Accuracy Prediction**: Uses advanced machine learning algorithms with high prediction accuracy
- **Multi-dimensional Analysis**: Supports analysis across multiple brands, cities, and time dimensions
- **Practical Application**: Designed specifically for tobacco industry characteristics
- **Scalability**: Modular design for easy extension and maintenance

## Key Algorithms

### XGBoost Configuration
```python
XGBRegressor(
    n_estimators=60,
    learning_rate=0.1,
    min_samples_leaf=2,
    max_depth=5,
    random_state=rnd_num
)
```

### Feature Engineering
- **Historical Features**: H1-H5 historical data points
- **Moving Averages**: 2-month, 3-month, 4-month, 5-month averages
- **Trend Features**: Slope calculations
- **Statistical Features**: Standard deviation
- **Business Features**: Order fulfillment rate, monthly ratios

## Performance Metrics

- **Standard Error**: Root Mean Square Error (RMSE)
- **R² Score**: Coefficient of determination
- **Accuracy Rate**: Percentage accuracy of predictions
- **Distribution Analysis**: Histogram of prediction accuracy

## Data Processing Pipeline

1. **Data Loading**: Load CSV files with sales data
2. **Data Cleaning**: Handle missing values and outliers
3. **Feature Calculation**: Compute business metrics and statistical features
4. **Time Series Processing**: Generate temporal features
5. **Model Training**: Train XGBoost model on historical data
6. **Prediction**: Generate predictions for target periods
7. **Evaluation**: Calculate accuracy metrics and generate reports

## Changelog

- **December 2024**: Initial project release
- **November 2024**: Completed XGBoost prediction model development
- **October 2024**: Completed data processing module development

## Note
This is part of the project code I wrote during my internship. It contains the main part of the project, but due to company reasons, the complete project code and dataset were not released.

---

*This project is a tobacco sales prediction research project for learning and research purposes only.* 