import pandas as pd
import numpy as np

df = pd.read_csv('raw_data.csv')
numerical_columns = ['Temperature', 'Humidity', 'Wind Speed']

means = df[numerical_columns].mean(axis=0)
stds = df[numerical_columns].std(axis=0)

df[numerical_columns] = (df[numerical_columns] - means) / stds

conditions = df['Weather Condition'].unique()
condition_to_label = {condition: idx for idx, condition in enumerate(conditions)}

df['condition_encoded'] = df['Weather Condition'].map(condition_to_label)
df.to_csv('processed_data.csv', index=False)
