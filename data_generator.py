import pandas as pd
import numpy as np
import random

def generate_data(n=5000):
    data = []

    for _ in range(n):
        amount = round(random.uniform(10, 50000), 2)
        hour = random.randint(0, 23)
        location_change = random.choice([0, 1])   # 1 = different city
        device_change = random.choice([0, 1])     # 1 = new device
        txn_count_24h = random.randint(1, 15)

        # FRAUD LOGIC (human-like)
        fraud = 0
        if amount > 20000 and location_change == 1:
            fraud = 1
        if txn_count_24h > 10 and device_change == 1:
            fraud = 1
        if hour < 5 and amount > 15000:
            fraud = 1

        data.append([
            amount,
            hour,
            location_change,
            device_change,
            txn_count_24h,
            fraud
        ])

    df = pd.DataFrame(data, columns=[
        "amount",
        "hour",
        "location_change",
        "device_change",
        "txn_count_24h",
        "fraud"
    ])

    return df


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("transactions.csv", index=False)
    print("transactions.csv generated successfully")
