import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.title("Portfolio Analysis & Forecasting App")
st.write("Select the tickers (companies) and corresponding weights that you would like to use in your investment portfolio.")
st.write("Popular tickers include: Apple=AAPL, Microsoft=MSFT, Google=GOOGL, TESLA=TSLA, Amazon=AMZN, Meta=META, Intel=INTC, Reddit=RDDT, Robloc=RBLX")
# User inputs
tickers_input = st.text_input("Enter tickers separated by commas", "TSLA,AAPL,GOOGL")
weights_input = st.text_input("Enter weights separated by commas (should sum to 1)", "0.33,0.33,0.34")

# Process inputs
tickers = [t.strip().upper() for t in tickers_input.split(',')]
weights = np.array([float(w.strip()) for w in weights_input.split(',')])

if len(tickers) != len(weights):
    st.error("Number of tickers and weights must match.")
elif not np.isclose(weights.sum(), 1):
    st.error("Weights must sum to 1.")
else:
    # Download data
    df = yf.download(tickers=tickers, start="2010-01-01")
    st.write("Downloaded data preview:")
    st.dataframe(df.head())

    close_df = df['Close']
    df['Portfolio Close'] = close_df.dot(weights)

    portfolio_label = " + ".join(f"{Fraction(w).limit_denominator()}{t}" for w, t in zip(weights, tickers))

    # Plot portfolio close price
    st.subheader("Daily Portfolio Close Prices (to Date)")
    fig, ax = plt.subplots(figsize=(12,6))
    df['Portfolio Close'].plot(ax=ax, color='green')
    ax.set_title("Total Closing Value of Portfolio")
    ax.text(pd.Timestamp('2012-02-01'), df['Portfolio Close'].max()-10,
            f"Portfolio = {portfolio_label}", bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    ax.set_ylabel("Portfolio Close Price")
    st.pyplot(fig)

    # Create lag features for forecasting
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Portfolio Close'].shift(lag)
    df = df.dropna()

    X = df[[f'lag_{lag}' for lag in range(1, 6)]]
    y = df['Portfolio Close']

    split_date = '2025-02-01'
    X_train = X[X.index < split_date]
    y_train = y[y.index < split_date]
    X_test = X[X.index >= split_date]
    y_test = y[y.index >= split_date]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"Test RMSE: {rmse:.2f}")

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted Portfolio Close")
    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.plot(y_test.index, y_test, label='Actual Portfolio Close')
    ax2.plot(y_test.index, y_pred, label='Predicted Portfolio Close')
    ax2.set_title('Portfolio Value Prediction with Random Forest')
    ax2.set_ylabel('Total Close Value')
    ax2.text(pd.Timestamp('2025-05-01'), y_test.max()-10,
             f"Portfolio = {portfolio_label}", bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    ax2.legend()
    st.pyplot(fig2)
    st.write("The predicted prices in the above plot are based on the true past 5 days' portfolio prices from the test set (One-Step-Ahead Prediction, sliding window of lag inputs)")

    # Quarterly Predictions
    quarterly_close = df['Portfolio Close'].resample('Q').last()  # or .mean() or .last(), choose what fits best

    st.subheader("Quarterly Portfolio Close Prices (to date)")
    st.line_chart(quarterly_close)

   # Prepare features/targets for multi-step forecasting
    n_lags = 5
    n_steps = 4

    q_df = pd.DataFrame(quarterly_close)

    for lag in range(1, n_lags + 1):
        q_df[f'lag_{lag}'] = q_df['Portfolio Close'].shift(lag)

    for step in range(1, n_steps + 1):
        q_df[f'target_{step}'] = q_df['Portfolio Close'].shift(-step)
    
    q_df = q_df[q_df.index > '2021-12-01']
    q_df = q_df.dropna()

    X = q_df[[f'lag_{lag}' for lag in range(1, n_lags + 1)]]
    y = q_df[[f'target_{step}' for step in range(1, n_steps + 1)]]

    # Get available quarterly dates for split
    quarter_dates = X.index.unique().sort_values()

    st.subheader("Data Range Used for Quarterly Predictions")
    st.write(quarter_dates.min(), "to", quarter_dates.max())

    split_date = pd.Timestamp('2024-06-30')
    st.write("Test/train split date:", split_date)


    X_train = X[X.index <= split_date]
    y_train = y[y.index <= split_date]
    X_test = X[X.index > split_date]
    y_test = y[y.index > split_date]

    st.write(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Train or test set is empty. Please choose a different split date.")
        st.stop()

    # Train model 
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    # Predict next 8 quarters 
    last_known = X_test.iloc[0].values.reshape(1, -1)
    predictions = model.predict(last_known).flatten()

    # Plot results 
    last_date = quarterly_close.index[-1]
    forecast_start = last_date + pd.offsets.QuarterEnd()
    forecast_index = pd.date_range(start=forecast_start, periods=n_steps, freq='Q')

    st.write("Forecast index dates:", forecast_index)

    forecast_series = pd.Series(predictions, index=forecast_index)

    import plotly.graph_objects as go

    # Combine historical and forecast data
    all_dates = quarterly_close.index.append(forecast_series.index)
    all_values = pd.concat([quarterly_close, forecast_series])

    fig = go.Figure()

    # Historical data trace
    fig.add_trace(go.Scatter(
        x=quarterly_close.index,
        y=quarterly_close.values,
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue')
    ))

    # Forecast data trace
    fig.add_trace(go.Scatter(
        x=forecast_series.index,
        y=forecast_series.values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='orange')
    ))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Portfolio Close Price',
        legend=dict(x=0.01, y=0.99)
    )
    st.subheader("Portfolio Quarterly Close Price: Historical and Forecast")
    st.plotly_chart(fig)
    
