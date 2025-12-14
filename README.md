# 📊 Zopper Device Insurance Attach % Analysis Dashboard

<div align="center">

**A comprehensive data analytics and visualization platform for analyzing device insurance attach percentages across retail stores and branches**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**🌐 [Live Application](https://zopper-cdhamqcmomscsdg3hribva.streamlit.app/)**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data Requirements](#-data-requirements)
- [Machine Learning Model](#-machine-learning-model)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

The **Zopper Device Insurance Attach % Analysis Dashboard** is an interactive web application designed to provide comprehensive insights into device insurance attach percentages across multiple retail stores and branches. Built with Streamlit and powered by machine learning, this dashboard enables data-driven decision-making through:

- **Real-time Analytics**: Interactive visualizations with black-themed UI for better data presentation
- **Predictive Analytics**: ML-powered forecasting for future attach percentages
- **Performance Categorization**: Automated classification of stores based on performance metrics
- **Multi-dimensional Analysis**: Branch-wise, store-wise, and monthly trend analysis

---

## ✨ Features

### 📊 Overview Dashboard
- **Executive Summary**: Key performance indicators at a glance
  - Total stores and branches count
  - Overall average attach percentage
  - Recent month performance metrics
- **Interactive Visualizations**: 
  - Monthly trend analysis with line charts
  - Branch performance comparison with bar charts
  - Complete dataset exploration

### 🏢 Branch Analysis
- Branch-specific performance metrics
- Monthly trend visualization for each branch
- Store performance ranking within branches
- Comparative analysis across all branches

### 🏪 Store Analysis
- Individual store performance tracking
- Monthly breakdown and trend analysis
- Best and worst performing months identification
- Performance comparison with average metrics

### 📅 Monthly Trends
- **Heatmap Visualization**: Comprehensive view of all stores across all months
- **Distribution Analysis**: Statistical distribution for each month
- **Top/Bottom Performers**: Identification of best and worst performing stores per month

### 🎯 Store Categorization
Automatic intelligent categorization of stores into performance tiers:
- **High Performer - Stable**: Mean ≥ 30%, Low volatility (< 10%)
- **High Performer - Volatile**: Mean ≥ 30%, High volatility (≥ 10%)
- **Medium Performer - Stable**: 20% ≤ Mean < 30%, Low volatility
- **Medium Performer - Volatile**: 20% ≤ Mean < 30%, High volatility
- **Low Performer**: 10% ≤ Mean < 20%
- **Underperformer - Critical**: Mean < 10%

### 🔮 January Prediction
- **Machine Learning Model**: Random Forest Regressor for accurate predictions
- **Model Performance Metrics**: MAE, RMSE, and R² Score
- **Store-level Predictions**: Individual predictions for each store
- **Branch-wise Aggregations**: Average predictions by branch
- **Export Functionality**: Download predictions as CSV

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| **Framework** | Streamlit 1.28+ |
| **Data Processing** | Pandas 2.0+, NumPy 1.24+ |
| **Visualization** | Plotly 5.0+ |
| **Machine Learning** | Scikit-learn 1.3+ |
| **Data I/O** | openpyxl 3.0+, xlrd 2.0+ |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sankalp250/Zopper.git
   cd Zopper
   ```

2. **Create a virtual environment** (Recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   streamlit --version
   ```

---

## 📖 Usage

### 🌐 Live Application

**Access the live dashboard directly**: [https://zopper-cdhamqcmomscsdg3hribva.streamlit.app/](https://zopper-cdhamqcmomscsdg3hribva.streamlit.app/)

The application is hosted on Streamlit Cloud and ready to use!

### Running Locally

1. **Ensure data file is present**
   - Place `Jumbo & Company_ Attach % .xls` in the project root directory

2. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Access the dashboard**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will automatically load and process the data

### Navigation Guide

- **Sidebar Menu**: Use the sidebar to switch between different analysis views
- **Interactive Charts**: Hover over charts for detailed information
- **Filters**: Use dropdown menus to filter data by branch, store, or month
- **Export**: Download predictions and reports as CSV files

### Key Workflows

1. **Quick Overview**: Start with the Overview Dashboard for executive summary
2. **Deep Dive**: Use Branch Analysis to understand regional performance
3. **Store Focus**: Navigate to Store Analysis for individual store insights
4. **Trend Analysis**: Check Monthly Trends for temporal patterns
5. **Performance Classification**: Review Store Categorization for strategic insights
6. **Forecasting**: Use January Prediction for future planning

---

## 📁 Project Structure

```
Zopper/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── .gitignore                        # Git ignore rules
├── Jumbo & Company_ Attach % .xls    # Input data file (Excel)
└── venv/                              # Virtual environment (not tracked)
```

### Code Organization

- **Data Loading**: `load_data()` - Handles Excel file reading and preprocessing
- **Store Categorization**: `categorize_stores()` - Performance-based classification
- **ML Model**: `train_prediction_model()` - Random Forest model training
- **Predictions**: `predict_january()` - Future month forecasting
- **Main App**: `main()` - Streamlit application entry point

---

## 📊 Data Requirements

### Input File Format

The application expects an Excel file (`Jumbo & Company_ Attach % .xls`) with the following structure:

| Column | Description | Format | Example |
|--------|-------------|--------|---------|
| `Branch` | Branch location | String | "Delhi_Ncr", "Pune", "Gujarat" |
| `Store_Name` | Store identifier | String | "Store_001", "Store_ABC" |
| `Aug` | August attach % | Decimal (0-1) | 0.15 (represents 15%) |
| `Sep` | September attach % | Decimal (0-1) | 0.18 |
| `Oct` | October attach % | Decimal (0-1) | 0.20 |
| `Nov` | November attach % | Decimal (0-1) | 0.22 |
| `Dec` | December attach % | Decimal (0-1) | 0.25 |

### Data Processing Notes

- Attach percentages are stored as decimals (0-1 range) in the Excel file
- The application automatically converts them to percentages (0-100%) for display
- Missing values should be handled before data import
- Data is cached for improved performance using Streamlit's caching mechanism

---

## 🤖 Machine Learning Model

### Model Architecture

- **Algorithm**: Random Forest Regressor
- **Ensemble Method**: 100 decision trees
- **Max Depth**: 10 levels
- **Random State**: 42 (for reproducibility)

### Feature Engineering

The model uses the following features:
- Historical monthly attach percentages (Sep, Oct, Nov, Dec)
- Mean attach percentage across months
- Standard deviation (volatility measure)
- Trend calculation (monthly change rate)

### Model Performance Metrics

- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors
- **R² Score**: Coefficient of determination (model fit quality)

### Training Process

1. Data is split into training (80%) and testing (20%) sets
2. Model is trained on historical patterns
3. Predictions are validated on test set
4. Model metrics are displayed for transparency

---

## 📸 Screenshots

> **Note**: Screenshots can be added here to showcase the dashboard interface

### Dashboard Views
- Overview Dashboard with executive summary
- Interactive charts with black-themed UI
- Branch and store performance visualizations
- Prediction interface with model metrics

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Jumbo & Company_ Attach % .xls`
- **Solution**: Ensure the Excel file is in the project root directory with the exact filename

**Issue**: `ModuleNotFoundError`
- **Solution**: Activate virtual environment and run `pip install -r requirements.txt`

**Issue**: Port already in use
- **Solution**: Use `streamlit run app.py --server.port 8502` to specify a different port

**Issue**: Data loading errors
- **Solution**: Verify Excel file format matches the required structure

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Test changes locally before submitting
- Update documentation for new features

---

## 📄 License

This project is created for assignment purposes. All rights reserved.

---

## 👤 Author

**Sankalp**

- GitHub: [@sankalp250](https://github.com/sankalp250)
- Project: [Zopper Dashboard](https://github.com/sankalp250/Zopper)

---

## 🙏 Acknowledgments

- Built for Zopper Data Science Internship Assignment
- Powered by Streamlit and Plotly for interactive visualizations
- Machine learning powered by Scikit-learn

---

<div align="center">

**Made with ❤️ for Data-Driven Decision Making**

⭐ Star this repo if you find it helpful!

</div>
