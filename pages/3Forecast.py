import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime as dt
from math import sqrt
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")
import logging
import math
import openai

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
# Azure OpenAI API client setup
openai.api_key = "bd18995c51fa40e19e493df21c7ded81"
openai.api_base = "https://madhukar-kumar.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

def get_completion(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo-16k",  # Replace with your actual model deployment name in Azure
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        logger.info("Successfully retrieved completion from OpenAI.")
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"Error connecting to OpenAI API: {e}")
        st.error("Error connecting to the OpenAI API. Please check your network connection.")
        return ""

# Function to generate insights based on DataFrame or plot description
def generate_insight(description: str) -> str:
    prompt = f"Provide a summary or insight in bullet points based on the following data or graph description: {description}"
    return get_completion(prompt)

st.set_page_config(
    page_title="Sales Forecast",
    layout='wide'
)

saletab, demandtab = st.tabs(['Sales Forecast','Demand Forecast'])
# App title
with saletab:
    st.title('Sales Forecasting')
    sales = None
    if 'data4' not in st.session_state:
        if st.button("Upload Dataset for Sales"):
            st.switch_page("data_upload.py")
    else:
        sales = st.session_state.data4

    if sales is not None:    
        sales_data = sales
        SaleDate = st.selectbox("Select column which contains Sales Date", sales.columns, index =None)
        sale_amt = st.selectbox("Select column which contains Sales Amount", sales.columns, index =None)
        if all([sales_data is not None, sale_amt is not None]):

            # Convert SalesDate to datetime format and set as index
            sales_data[SaleDate] = pd.to_datetime(sales_data[SaleDate]).dt.date
            sales_data = sales_data.groupby(SaleDate)[sale_amt].sum().reset_index()

            # Data overview section
            st.header('Dataset Overview')
            st.write('Preview of the Sales Data:')
            st.write(sales_data.head())

            # Select forecasting period
            st.header('Forecasting Period')
            forecast_period = st.slider('Select the number of days to forecast:', min_value=2, max_value=10, value=5, step=1)

            # Train-Test Split
            # st.header('Train-Test Split')
            sales_data = sales_data.set_index(SaleDate)
            train_size = 80
            # st.slider('Select the percentage of data to use for training:', min_value=50, max_value=98, value=80, step=1)

            y = sales_data[sale_amt]


            # Model selection
            st.header('Model Selection')
            model_choice = st.selectbox(
                'Select a machine learning model:',
                ['ARIMA', 'Prophet']
            )

            # Split the data into training and testing sets
            split_index = int(len(sales_data) * train_size / 100)
            y_train, y_test = y[:split_index], y[split_index:]


            # Fit the ARIMA model 
            if model_choice == 'ARIMA':
                try:
                    from pmdarima import auto_arima
                    stepwise_fit = auto_arima(y, trace=True, suppress_warnings=True)
                    p = stepwise_fit.order[0]
                    d = stepwise_fit.order[1]
                    q = stepwise_fit.order[2]

                    model = ARIMA(y_train, order = (p,d,q))
                    model_fit = model.fit()
                    # Forecast the test set and future periods
                    y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

                    future_dates = pd.date_range(start = sales_data.index[-1], periods=forecast_period+1, freq='D')[1:]
                    y_future_pred = model_fit.predict(start=len(sales_data), end=len(sales_data)+forecast_period-1)

                    # Performance metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"Mean Squared Error (MSE) on Test Data: {mse:.2f}")
                    st.write(f"Root Mean Squared Error (RMSE) on Test Data: {rmse:.2f}")
                    st.write(f"Mean Absolute Error (MAE) on Test Data: {mae:.2f}")
                    st.write(f"R2-Score on Test Data: {r2:.2f}")


                    # Plotting actual vs predicted sales dollars (test set)
                    st.header('Actual vs Predicted Sales Dollars (Test Set)')
                    fig, ax = plt.subplots(figsize=(12,4))
                    ax.plot(y_test.index, y_test, label='Actual Sales Dollars', color='blue')
                    ax.plot(y_test.index, y_pred, label='Predicted Sales Dollars', color='red', linestyle='--')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sales Dollars')
                    ax.set_title('Actual vs Predicted Sales Dollars (Test Set)')
                    ax.legend()
                    st.pyplot(fig)

                    # Plotting future forecast
                    st.header('Future Sales Dollars Forecast')
                    fig, ax = plt.subplots(figsize=(12,4))
                    ax.plot(future_dates, y_future_pred, label='Forecasted Sales Dollars', color='green')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sales Dollars')
                    ax.set_title(f'Forecasted Sales Dollars for {forecast_period} Days')
                    ax.legend()
                    st.pyplot(fig)
                    df_description = y_future_pred
                    insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
                    st.write("Insights for forecasted sales dollars:")
                    st.write(insight)
                except Exception as e:
                    logger.error("Error during Arima:%s",e)
                    st.error("Error during performing ARIMA")


            else:
                try:
                    sales_data = sales_data.rename(columns ={SaleDate:'ds',
                                                        sale_amt:'y'})
                    model = Prophet()
                    model_fit = model.fit(sales_data[['ds', 'y']])
                    future = model.make_future_dataframe(periods=12)
                    forecast = model.predict(future)
                    

                    st.write(f"Forecasted Sales for the next {forecast_period} days:")
                    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period))

                    

                    # Plot the forecast
                    st.write("Sales Dollar Forecast:")
                    fig1 = model.plot(forecast)
                    st.pyplot(fig1)

                    # Plot forecast components
                    st.write("Forecast Components (Trend, Yearly Seasonality):")
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)

                    # Plot the actual vs predicted values for historical data
                    st.write("Actual vs Predicted Sales Dollars (Historical Data):")
                    forecasted_values = forecast.set_index('ds').join(sales_data.set_index('ds'), rsuffix='_actual')

                    # Calculate and display error metrics (on historical data)
                    mse = mean_squared_error(forecasted_values.dropna()['y'], forecasted_values.dropna()['yhat'])
                    mae = mean_absolute_error(forecasted_values.dropna()['y'], forecasted_values.dropna()['yhat'])
                    
                    # Calculate R² score (on historical data)
                    r2 = r2_score(forecasted_values.dropna()['y'], forecasted_values.dropna()['yhat'])
                    
                    st.write(f"Mean Squared Error (MSE) on Historical Data: {mse:.2f}")
                    st.write(f"Mean Absolute Error (MAE) on Historical Data: {mae:.2f}")
                    st.write(f"R² Score on Historical Data: {r2:.2f}")
                    
                    fig3, ax = plt.subplots()
                    ax.plot(forecasted_values.index, forecasted_values['y'], label='Actual Sales', color='blue')
                    ax.plot(forecasted_values.index, forecasted_values['yhat'], label='Predicted Sales', color='red', linestyle='--')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sales Dollars')
                    ax.set_title('Actual vs Predicted Sales Dollars')
                    ax.legend()
                    st.pyplot(fig3)

                except Exception as e:
                    logging.error("An error occurred during Prophet: %s", e)
                    st.error("An error occurred while performing Prophet.")
                

with demandtab:
    st.title("Demand Forecasting")  
    sales = None
    if 'data4' not in st.session_state:
        if st.button("Upload Sales Dataset for Demand forecasting"):
            st.switch_page("data_upload.py")
    else:
        sales = st.session_state.data4

    if sales is not None:
        prdct = st.selectbox("Select the column which contains the product", sales.columns, index= None)
        SaleDate = st.selectbox("Select column which contain Sales Date", sales.columns, index =None)
        Qnt = st.selectbox("Select column which contains Quantity", sales.columns, index =None)

        if all([prdct is not None, SaleDate is not None, Qnt is not None]):
            slctprdct = st.selectbox("Select a product for demand forecasting", sales[prdct].unique(), index= None )
            sales[SaleDate] = pd.to_datetime(sales[SaleDate])
            if slctprdct is not None:
                sales_data = sales[sales[prdct]==slctprdct]
                start_date = sales[SaleDate].min()
                end_data = sales[SaleDate].max()
                df_agg = sales_data.groupby([SaleDate]).agg({Qnt: 'sum'}).reset_index()
                all_dates = pd.date_range(start=start_date, end=end_data)
                df_complete = pd.DataFrame({SaleDate:all_dates})  
                df_forecast = pd.merge(df_complete, df_agg, on=[SaleDate] ,how='left').fillna(0)
                df_forecast[SaleDate]= df_forecast[SaleDate].dt.date
                st.dataframe(df_forecast.tail())

                # Select forecasting period
                st.header('Forecasting Period')
                forecast_period = st.slider('Select the number of days to forecast Demand:', min_value=2, max_value=10, value=5, step=1)
                
                df_forecast.set_index(SaleDate, inplace=True)
                
                y = df_forecast[Qnt]
                # # Split the data into training and testing sets
                # train_size = 80
                # split_index = int(len(sales_data) * train_size / 100)
                y_train, y_test = y[:-10], y[-10:]

                import statsmodels.api as sm
                from pmdarima import auto_arima
                stepwise_fit = auto_arima(y, trace=True, suppress_warnings=True)
                p = stepwise_fit.order[0]
                d = stepwise_fit.order[1]
                q = stepwise_fit.order[2]
                
                # st.write(p,d,q)

                # Function to perform ARIMA forecasting
                def arima_forecast(product_data, forecast_period):
                    # product_data.set_index(SaleDate, inplace=True)
                    # product_data = product_data.resample('D').sum()  # Resample to daily frequency
                    # product_data = product_data.fillna(0)  # Fill missing values with 0
                    model = sm.tsa.ARIMA(product_data[Qnt], order=(p,d,q))  # ARIMA(1, 1, 1)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps = forecast_period)  # Forecast the next 30 days
                    return forecast

                # # Forecast for each product
                forecast_results = {}
                # for product in Product_IDs:
                #     product_data = df_forecast[df_forecast['Product_ID'] == product]
                forecast_results[f"Demand Forecast for {slctprdct}"] = arima_forecast(df_forecast, forecast_period)
                result = pd.DataFrame(forecast_results).reset_index(names=['Date'])
                result["Date"] = result['Date'].dt.date
                result[f"Demand Forecast for {slctprdct}"]= result[f"Demand Forecast for {slctprdct}"].apply(math.ceil)

                st.dataframe(result)

                # # Display forecast results
                # for product, forecast in forecast_results.items():
                #     print(f"Forecast for {product}:")
                #     print(forecast.head())

                # # Plotting actual vs predicted sales dollars (test set)
                # st.header('Actual vs Predicted Sales Dollars (Test Set)')
                # fig, ax = plt.subplots(figsize=(12,4))
                # ax.plot(y_test.index, y_test, label='Actual Sales Dollars', color='blue')
                # ax.plot(y_test.index, y_pred, label='Predicted Sales Dollars', color='red', linestyle='--')
                # ax.set_xlabel('Date')
                # ax.set_ylabel('Sales Dollars')
                # ax.set_title('Actual vs Predicted Sales Dollars (Test Set)')
                # ax.legend()
                # st.pyplot(fig)
                model = ARIMA(y_train, order = (p,d,q))
                model_fit = model.fit()
                # Forecast the test set and future periods
                y_pred = model_fit.forecast(steps = 10)

                future_dates = pd.date_range(start = sales_data.index[-1], periods=forecast_period+1, freq='D')[1:]
                y_future_pred = model_fit.predict(start=len(sales_data), end=len(sales_data)+forecast_period-1)

                # Performance metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Mean Squared Error (MSE) on Test Data: {mse:.2f}")
                st.write(f"Root Mean Squared Error (RMSE) on Test Data: {rmse:.2f}")
                st.write(f"Mean Absolute Error (MAE) on Test Data: {mae:.2f}")
                st.write(f"R2-Score on Test Data: {r2:.2f}")


                # Plotting actual vs predicted sales dollars (test set)
                st.header('Actual vs Predicted Sales Dollars (Test Set)')
                fig, ax = plt.subplots(figsize=(12,4))
                ax.plot(y_test.index, y_test, label='Actual Sales Dollars', color='blue')
                ax.plot(y_test.index, y_pred, label='Predicted Sales Dollars', color='red', linestyle='--')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales Dollars')
                ax.set_title('Actual vs Predicted Sales Dollars (Test Set)')
                ax.legend()
                st.pyplot(fig)

                # Plotting future forecast
                st.header('Future Demand Forecast')
                fig, ax = plt.subplots(figsize=(12,4))
                ax.plot(result['Date'], result[f"Demand Forecast for {slctprdct}"], label=f"Demand Forecast for {slctprdct}", color='green')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales Dollars')
                ax.set_title(f'Forecasted demand of {slctprdct} for {forecast_period} Days')
                ax.legend()
                st.pyplot(fig)
                df_description = result
                insight = generate_insight(f"This is a summary of the following DataFrame:\n{df_description}")
                st.write(f"Insights for forecasted {slctprdct} demand:")
                st.write(insight)

    
    
    
    
    







