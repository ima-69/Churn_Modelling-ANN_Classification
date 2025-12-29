# 🏦 Customer Churn Prediction - ANN Classification

An AI-powered web application that predicts customer churn probability using Deep Learning (Artificial Neural Networks). Built with TensorFlow/Keras and deployed with a modern Streamlit interface.

## 📊 Project Overview

This project uses a deep learning model to predict whether a bank customer is likely to leave (churn) based on their profile and account information. The model analyzes various features including demographics, account details, and customer behavior to provide accurate churn predictions.

## ✨ Features

- **Modern Web Interface**: User-friendly Streamlit dashboard with gradient designs
- **Real-time Predictions**: Instant churn probability calculation
- **Visual Results**: Color-coded risk indicators (green for low risk, red for high risk)
- **Comprehensive Input Fields**: 
  - Customer demographics (Geography, Gender, Age)
  - Account details (Credit Score, Balance, Salary)
  - Banking relationship (Tenure, Products, Activity status)

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep Learning framework
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Data preprocessing (StandardScaler, LabelEncoder, OneHotEncoder)

## 📁 Project Structure

```
Churn_Modelling-ANN_Classification/
│
├── app.py                              # Streamlit web application
├── style.css                           # Custom CSS styling
├── experiments.ipynb                   # Model training notebook
├── prediction.ipynb                    # Prediction testing notebook
├── model.h5                            # Trained ANN model
├── Churn_Modelling.csv                # Dataset
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── Pickle files/
│   ├── label_encoder_gender.pkl       # Gender encoder
│   ├── one_hot_encoder_geography.pkl  # Geography encoder
│   └── scaler.pkl                     # Feature scaler
│
└── logs/                              # TensorBoard logs
    └── fit/
```

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd Churn_Modelling-ANN_Classification
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

## 📊 Model Details

- **Architecture**: Artificial Neural Network (ANN)
- **Input Features**: 10 features including CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **Output**: Binary classification (Churn probability: 0-1)
- **Training Data**: Banking customer dataset with historical churn information

## 🎯 Usage

1. **Launch the application** using `streamlit run app.py`
2. **Enter customer information**:
   - Select geography and gender
   - Adjust sliders for age, tenure, and number of products
   - Input credit score, balance, and salary
   - Indicate credit card ownership and member activity status
3. **Click "Predict Churn"** button
4. **View results**: The app displays churn probability with color-coded risk assessment

## 📈 Model Training

To retrain the model, open and run `experiments.ipynb`:
- Data preprocessing and feature engineering
- Model architecture definition
- Training with TensorBoard logging
- Model evaluation and saving

## 🧪 Testing Predictions

Use `prediction.ipynb` to test individual predictions and validate model performance.

## 📦 Requirements

```
tensorflow>=2.10.0
streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
```

## 🎨 Interface Features

- **Responsive Design**: Optimized for laptop screens
- **Gradient Buttons**: Modern purple gradient with hover effects
- **Color-Coded Results**: 
  - 🟢 Green gradient for low churn risk
  - 🔴 Red gradient for high churn risk
- **Clean Layout**: Two-column input sections for better organization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👨‍💻 Author

**Imansha Dilshan**

© 2025 Imansha Dilshan. All rights reserved.

## 🙏 Acknowledgments

- Dataset source: Bank customer churn data
- Built with TensorFlow and Streamlit
- Inspired by real-world customer retention challenges

---

**Note**: This is an educational project demonstrating the application of deep learning for customer churn prediction.