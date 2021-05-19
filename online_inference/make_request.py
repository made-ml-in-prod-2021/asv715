"""
Make requests to service to get model predictions
"""
import requests
import numpy as np
import pandas as pd

REQUEST_URL = "http://0.0.0.0:80/predict"

if __name__ == "__main__":
    data = pd.read_csv("data/test.csv")
    request_features = list(data.columns)

    for i in range(1):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]

        response = requests.get(
            REQUEST_URL,
            json={
                "data": request_data,
                "features": request_features
            },
        )

        if response.status_code == 200:
            print(response.json())
        elif response.status_code == 400:
            print('Validation error')
        else:
            print('Something went wrong')
