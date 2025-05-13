import pandas as pd

def import_historical_from_csv(csv_path):
    print("Імпортуємо історичні дані з CSV...")
    with open(csv_path, encoding='utf-8') as f:
        lines = f.readlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('2020'):
            data_start = i
            break
    if data_start is None:
        raise ValueError('Не знайдено початок даних у CSV!')
    df = pd.read_csv(csv_path, skiprows=data_start)
    df = df.rename(columns={df.columns[0]: 'date', df.columns[3]: 'temperature'})
    df = df[['date', 'temperature']]
    df['date'] = pd.to_datetime(df['date'].str[:8], format='%Y%m%d')
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df = df.dropna()
    print("Історичні дані завантажено у пам'ять")
    return df

if __name__ == "__main__":
    import_historical_from_csv('path_to_your_csv.csv') 