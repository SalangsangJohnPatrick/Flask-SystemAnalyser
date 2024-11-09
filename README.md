# System Analyser: Web Application for Predicting the Estimated Relative Performance of Computer Hardwares using Multiple Linear Regression

A web-based application built with Flask that performs statistical analysis and visualization of CSV data using linear regression. The app allows users to upload CSV files, select dependent and independent variables, and generates statistical insights along with visualization plots.

## Features

- CSV file upload and parsing
- Linear regression analysis with multiple independent variables
- Statistical metrics calculation:
  - Standard error
  - Correlation coefficient
  - Correlation strength description
- Automated visualization generation:
  - Scatter plots with best fit lines
  - Individual variable analysis
- Interactive web interface
- Real-time data processing

## Prerequisites

```bash
python >= 3.6
pip
```

## Required Dependencies 

```bash
Flask
numpy
pandas
matplotlib
scikit-learn
```

## Usage

1. **Home Page** (`/`):
   - Upload CSV files
   - View imported data in tabular format

2. **Data Analysis** (`/analyze_data`):
   - Select dependent variable
   - Choose one or more independent variables
   - Submit for analysis

3. **Results**:
   - View regression analysis results
   - Examine correlation statistics
   - Visualize relationships through plots

## Technical Details

### Linear Regression Analysis
The application uses scikit-learn's LinearRegression model to:
- Fit data to the linear model
- Generate predictions
- Calculate coefficients and intercept

### Statistical Calculations
- **Standard Error**: Calculated using residuals
- **Correlation Coefficient**: Computed using numpy's corrcoef
- **Correlation Description**: Automatically categorized based on coefficient strength

### Visualization
- Generated using matplotlib
- Includes:
  - Scatter plots of actual data points
  - Best fit regression lines
  - Proper labeling and legends
- Plots are converted to base64 for web display
