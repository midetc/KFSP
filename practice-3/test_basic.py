import unittest
import os
import pandas as pd
from weather_forecast.data_loader import import_historical_from_csv
from weather_forecast.weather_predictor import WeatherPredictor
from weather_forecast.db_utils import insert_forecast, init_db, get_db_path

CSV_PATH = os.path.join(os.path.dirname(__file__), '../dataexport_20250512T113413.csv')

class TestWeatherForecastMinimal(unittest.TestCase):
    def setUp(self):
        self.df = import_historical_from_csv(CSV_PATH)
        self.predictor = WeatherPredictor()
        self.predictor.set_data(self.df)

    def test_import_csv(self):
        self.assertFalse(self.df.empty)
        self.assertIn('temperature', self.df.columns)
        self.assertIn('date', self.df.columns)

    def test_predict(self):
        forecast = self.predictor.predict(self.df['date'].max() + pd.Timedelta(days=1), days=7)
        self.assertEqual(len(forecast), 7)
        self.assertIn('min_temp', forecast.columns)
        self.assertIn('max_temp', forecast.columns)

    def test_sanity_check(self):
        forecast = self.predictor.predict(self.df['date'].max() + pd.Timedelta(days=1), days=7)
        valid, msg = self.predictor.sanity_check(forecast)
        self.assertTrue(valid)

    def test_insert_forecast(self):
        init_db()
        forecast = self.predictor.predict(self.df['date'].max() + pd.Timedelta(days=1), days=3)
        insert_forecast(forecast)
        self.assertTrue(os.path.exists(get_db_path()))

if __name__ == '__main__':
    unittest.main() 