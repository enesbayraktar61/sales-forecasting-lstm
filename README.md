# Sales Forecasting (LSTM)

This project builds a deep learning model to forecast product sales using time series data and LSTM networks.

The model was trained using TensorFlow/Keras and deployed with Streamlit on Hugging Face Spaces.

---

## Project Overview

- Problem Type: Time Series Forecasting  
- Approach: Deep Learning (LSTM)  
- Framework: TensorFlow / Keras  
- Deployment: Streamlit (Hugging Face Spaces)

---

## Dataset

The dataset contains historical daily sales records for multiple stores and items.

For modeling, a single time series was selected:

- Store: store_1  
- Item: item_1  

This simplifies the forecasting setup and allows clearer modeling of temporal patterns.

---

## Data Preprocessing

### Time Series Preparation

- Converted `date` column to datetime format  
- Sorted observations chronologically  
- Set date as index  

### Feature Selection

For the baseline LSTM model:

- Used only: `sales` (target variable)

Future improvements may include:

- price  
- promo  
- weekday  

### Scaling

- Applied MinMaxScaler to normalize values between 0 and 1  
- Required for stable LSTM training  

### Sequence Creation

- Sequence length: 30 days  
- Each input window predicts the next day’s sales  

---

## Modeling

### LSTM Architecture

The model includes:

- LSTM layer (50 units, return_sequences=True)  
- LSTM layer (50 units)  
- Dense output layer  

Loss function:

- Mean Squared Error (MSE)

Optimizer:

- Adam

---

## Results

The model successfully captured overall demand trends while smoothing noisy fluctuations typical in retail sales data.

Key observations:

- Predictions follow the overall trend  
- Short-term noise is smoothed  
- No major overfitting observed  

This demonstrates strong performance for a baseline deep learning forecasting model.

---

## Deployment

The trained model was saved in `.keras` format and deployed using Streamlit.

The app allows users to:

- Input the last 30 sales values  
- Predict the next day’s sales  

---

## Conclusion

This project demonstrates how LSTM networks can effectively model time series data for business forecasting.

By combining proper preprocessing, sequence modeling, and deep learning techniques, the model produces stable and realistic forecasts useful for inventory planning and demand prediction.

---

## Future Improvements

- Add exogenous features (price, promotions)  
- Tune hyperparameters  
- Try GRU / Transformer models  
- Multi-step forecasting  

---

## How to Run Locally

```bash
git clone https://github.com/enesbayraktar61/sales-forecasting-lstm.git
cd sales-forecasting-lstm
pip install -r requirements.txt
streamlit run app.py
