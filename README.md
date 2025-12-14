# Zopper Device Insurance Attach % Analysis

A comprehensive data analysis and visualization dashboard for analyzing device insurance attach percentages across retail stores.

## 📊 Overview

This project provides an in-depth analysis of device insurance attach percentages from Jumbo & Company retail stores. The analysis includes:

- **Branch-wise Performance Analysis**: Compare attach percentages across different branches
- **Store-level Insights**: Detailed analysis of individual store performance
- **Monthly Trends**: Track performance over time (Aug-Dec)
- **Store Categorization**: Automatic categorization of stores based on performance metrics
- **January Prediction**: Machine learning model to predict attach percentages for January 2026

## 🚀 Features

### 1. Overview Dashboard
- Executive summary with key metrics
- Overall monthly trends
- Branch performance comparison
- Complete dataset visualization

### 2. Branch Analysis
- Branch-wise performance metrics
- Monthly trends for each branch
- Store performance ranking within branches

### 3. Store Analysis
- Individual store performance tracking
- Monthly breakdown for each store
- Performance trends and comparisons

### 4. Monthly Trends
- Heatmap visualization of all stores across months
- Distribution analysis for each month
- Top and bottom performing stores

### 5. Store Categorization
- Automatic categorization into performance tiers:
  - High Performer - Stable
  - High Performer - Volatile
  - Medium Performer - Stable
  - Medium Performer - Volatile
  - Low Performer - Needs Improvement
  - Underperformer - Critical

### 6. January Prediction
- Machine learning model (Random Forest Regressor) to predict January attach %
- Model performance metrics
- Store-level predictions
- Branch-wise average predictions
- Downloadable predictions CSV

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/sankalp250/Zopper.git
cd Zopper
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

1. Ensure the Excel file `Jumbo & Company_ Attach % .xls` is in the project root directory

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

4. Use the sidebar to navigate between different analysis views

## 📁 Project Structure

```
Zopper/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── Jumbo & Company_ Attach % .xls    # Input data file
└── venv/                              # Virtual environment (not tracked)
```

## 🔍 Data Structure

The input Excel file should contain the following columns:
- **Branch**: Branch location (e.g., Delhi_Ncr, Pune, Gujarat)
- **Store_Name**: Name of the retail store
- **Aug, Sep, Oct, Nov, Dec**: Monthly attach percentages (values between 0-1)

## 📈 Key Insights

The dashboard provides comprehensive insights including:
- Overall attach percentage trends
- Branch performance comparisons
- Store-level performance tracking
- Identification of high and low performers
- Predictive analytics for future months

## 🤖 Machine Learning Model

The January prediction model uses:
- **Algorithm**: Random Forest Regressor
- **Features**: Historical monthly attach percentages, trends, volatility
- **Metrics**: MAE, RMSE, R² Score

## 📝 Notes

- Attach percentages in the data are stored as decimals (0-1 range) and are converted to percentages (0-100%) for display
- The prediction model is trained on historical patterns and may need retraining as new data becomes available
- Store categorization is based on mean attach percentage and standard deviation

## 👤 Author

Created for Zopper Data Science Internship Assignment

## 📄 License

This project is created for assignment purposes.

