from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# ------------------ TASK 1: MERGE DATA ------------------

def merge_data():
    import pandas as pd

    flights = pd.read_csv('/opt/airflow/dags/flights.csv')
    users = pd.read_csv('/opt/airflow/dags/users.csv')
    hotels = pd.read_csv('/opt/airflow/dags/hotels.csv')

    # Merge flights + users
    df = pd.merge(flights, users, left_on='userCode', right_on='code', how='left')

    # Merge with hotels (FIXED)
    df = pd.merge(df, hotels, on=['travelCode', 'userCode'], how='left')

    # Rename columns
    df.rename(columns={'price_x': 'price', 'price_y': 'hotel_price'}, inplace=True)

    # Handle missing values
    df.fillna(0, inplace=True)

    print("Merged Shape:", df.shape)

    df.to_csv('/opt/airflow/dags/merged.csv', index=False)


# ------------------ TASK 2: CLEAN + FEATURE ENGINEERING ------------------

def clean_and_engineer_data():
    import pandas as pd

    df = pd.read_csv('/opt/airflow/dags/merged.csv')

    df['route'] = df['from'] + '-' + df['to']

    def convert_time(time):
        hour = int(time)
        minute = time - hour
        return int(hour * 60) + int(minute * 100)

    df['total_time'] = df['time'].apply(convert_time)

    df['date_x'] = pd.to_datetime(df['date_x'])

    df['weekday_num'] = df['date_x'].dt.weekday
    df['month'] = df['date_x'].dt.month
    df['year'] = df['date_x'].dt.year

    df['speed'] = df['distance'] / (df['total_time'] / 60)

    # Handle missing again (safety)
    df.fillna(0, inplace=True)

    # Drop unnecessary columns
    df.drop(['travelCode', 'userCode', 'code', 'date_x', 'date_y', 'time'], axis=1, inplace=True)

    df.to_csv('/opt/airflow/dags/features.csv', index=False)


# ------------------ TASK 3: ENCODING ------------------

def encode_features():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import pickle

    df = pd.read_csv('/opt/airflow/dags/features.csv')

    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

    encoders = {}
    encoded_df = df.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    with open('/opt/airflow/dags/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    encoded_df.to_csv('/opt/airflow/dags/encoded.csv', index=False)


# ------------------ TASK 4: PREPARE DATA ------------------

def prepare_training_data():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import pickle

    df = pd.read_csv('/opt/airflow/dags/encoded.csv')

    X = df.drop('price', axis=1)
    y = df['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open('/opt/airflow/dags/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    pd.DataFrame(X_scaled).to_csv('/opt/airflow/dags/X_scaled.csv', index=False)
    y.to_csv('/opt/airflow/dags/y.csv', index=False)


# ------------------ TASK 5: SPLIT DATA ------------------

def split_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    X = pd.read_csv('/opt/airflow/dags/X_scaled.csv')
    y = pd.read_csv('/opt/airflow/dags/y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    X_train.to_csv('/opt/airflow/dags/X_train.csv', index=False)
    X_test.to_csv('/opt/airflow/dags/X_test.csv', index=False)
    y_train.to_csv('/opt/airflow/dags/y_train.csv', index=False)
    y_test.to_csv('/opt/airflow/dags/y_test.csv', index=False)


# ------------------ TASK 6: TRAIN MODEL ------------------

def train_model():
    import pandas as pd
    import pickle
    from sklearn.linear_model import LinearRegression

    X_train = pd.read_csv('/opt/airflow/dags/X_train.csv')
    y_train = pd.read_csv('/opt/airflow/dags/y_train.csv')

    # 🔥 Handle any leftover NaN (extra safety)
    X_train.fillna(0, inplace=True)
    y_train.fillna(0, inplace=True)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open('/opt/airflow/dags/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained successfully ✅")


# ------------------ DAG ------------------

with DAG(
    dag_id='flight_price_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    t1 = PythonOperator(task_id='merge_data', python_callable=merge_data)
    t2 = PythonOperator(task_id='clean_and_engineer', python_callable=clean_and_engineer_data)
    t3 = PythonOperator(task_id='encode_features', python_callable=encode_features)
    t4 = PythonOperator(task_id='prepare_data', python_callable=prepare_training_data)
    t5 = PythonOperator(task_id='split_data', python_callable=split_data)
    t6 = PythonOperator(task_id='train_model', python_callable=train_model)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6