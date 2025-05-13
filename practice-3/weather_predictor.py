import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

class WeatherPredictor:
    def __init__(self):
        self.data = None
        self.model = None

    def set_data(self, df):
        self.data = df.copy()
        prophet_df = self.data.rename(columns={
            'date': 'ds',
            'temperature': 'y'
        })
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.model.fit(prophet_df)

    def predict(self, target_date, days=30):
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        future_dates = pd.date_range(
            start=target_date,
            periods=days,
            freq='D'
        )
        future_df = pd.DataFrame({'ds': future_dates})
        
        forecast = self.model.predict(future_df)
        
        result = []
        for _, row in forecast.iterrows():
            daily_variation = np.random.normal(0, 1)
            min_temp = round(row['yhat_lower'] + daily_variation, 1)
            max_temp = round(row['yhat_upper'] + daily_variation, 1)
            
            if min_temp >= max_temp:
                min_temp, max_temp = max_temp - 0.1, min_temp + 0.1
                
            result.append({
                'date': row['ds'],
                'min_temp': min_temp,
                'max_temp': max_temp,
                'forecast': f"{min_temp}-{max_temp}°C"
            })
        
        return pd.DataFrame(result)

    def evaluate(self, test_data):
        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("Тестові дані мають бути DataFrame")
            
        test_df = test_data.rename(columns={
            'date': 'ds',
            'temperature': 'y'
        })
        
        forecast = self.model.predict(test_df[['ds']])
        
        mae = mean_absolute_error(test_df['y'], forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))
        
        within_range = ((test_df['y'] >= forecast['yhat_lower']) &
                       (test_df['y'] <= forecast['yhat_upper'])).mean() * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'within_range_percent': within_range
        }

    def plot_forecast(self, forecast_df):
        plt.figure(figsize=(15, 8))
        plt.fill_between(
            forecast_df['date'], 
            forecast_df['min_temp'], 
            forecast_df['max_temp'], 
            alpha=0.2, 
            color='skyblue'
        )
        plt.plot(forecast_df['date'], forecast_df['min_temp'], 'b-', label='Мін. температура')
        plt.plot(forecast_df['date'], forecast_df['max_temp'], 'r-', label='Макс. температура')
        plt.title('Прогноз погоди на рік')
        plt.xlabel('Дата')
        plt.ylabel('Температура (°C)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['min_temp'],
                fill=None,
                mode='lines',
                line_color='blue',
                name='Мін. температура'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['max_temp'],
                fill='tonexty',
                mode='lines',
                line_color='red',
                name='Макс. температура'
            ))
            
            fig.update_layout(
                title='Прогноз погоди на рік (інтерактивний)',
                xaxis_title='Дата',
                yaxis_title='Температура (°C)',
                hovermode='x unified',
                width=1000,
                height=600
            )
            
            fig.write_html(os.path.join(os.path.dirname(__file__), 'forecast_interactive.html'))
            print("Створено інтерактивний графік: forecast_interactive.html")
        except Exception as e:
            print(f"Помилка при створенні інтерактивного графіка: {e}")
        
        return plt

    def sanity_check(self, forecast_df):
        if forecast_df['min_temp'].min() < -30 or forecast_df['max_temp'].max() > 40:
            return False, "Температура в діапазоні нереалістичних значень"
            
        summer_dates = [d for d in forecast_df['date'] if d.month in [6, 7, 8]]
        winter_dates = [d for d in forecast_df['date'] if d.month in [12, 1, 2]]
        
        if summer_dates and winter_dates:
            summer_temps = forecast_df[forecast_df['date'].isin(summer_dates)]['max_temp'].mean()
            winter_temps = forecast_df[forecast_df['date'].isin(winter_dates)]['max_temp'].mean()
            if winter_temps >= summer_temps:
                return False, "Порушена сезонність: зимні температури вищі за літні"
                
        if (forecast_df['max_temp'] - forecast_df['min_temp']).min() <= 0:
            return False, "Мінімальна температура вище максимальної"
            
        return True, "Прогноз пройшов перевірку" 