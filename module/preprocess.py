import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(csv_path):
    """
    Preprocess the dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file containing the dataset.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two elements. The first element is a 2D array of features, and the second is a 1D array of targets.
    """
    # Load data
    data = pd.read_csv(csv_path)
    data = data.sample(n=10000)
    data = data.drop(['subregion', 'parent'], axis=1)
    data['time'] = pd.to_datetime(data['time'])

    # Extract features from 'time'
    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek
    data['day_of_month'] = data['time'].dt.day
    data.sort_values('time', inplace=True)

    # Drop original 'time' column
    data = data.drop(['time'], axis=1)

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = ['hour', 'day_of_week', 'day_of_month']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Separate features and target
    features = data.drop(['value'], axis=1).values
    target = data['value'].values

    return features, target