import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of sessions
num_sessions = 1000

# Generate session IDs (1 to 1000)
session_ids = np.arange(1, num_sessions + 1)

# Generate random user IDs between 1 and 300 (simulate repeat users)
user_ids = np.random.randint(1, 301, size=num_sessions)

# Generate random timestamps in the year 2022
base_date = datetime(2022, 1, 1)
random_days = np.random.randint(0, 365, size=num_sessions)
random_seconds = np.random.randint(0, 86400, size=num_sessions)
timestamps = [base_date + timedelta(days=int(d), seconds=int(s))
              for d, s in zip(random_days, random_seconds)]

# Define possible actions with probabilities
actions = np.random.choice(
    ['view', 'click', 'add_to_cart', 'purchase'],
    size=num_sessions,
    p=[0.5, 0.3, 0.15, 0.05]
)

# Generate random prices between $5 and $500, rounded to 2 decimals
prices = np.round(np.random.uniform(5, 500, size=num_sessions), 2)

# Define product categories
categories = np.random.choice(
    ['Electronics', 'Fashion', 'Home', 'Sports', 'Toys', 'Automotive'],
    size=num_sessions
)

# Generate purchase indicator:
# - If action is 'purchase': purchase = 1
# - Else, assign purchase=1 with probabilities depending on the action type
purchase = []
for act in actions:
    if act == 'purchase':
        purchase.append(1)
    elif act == 'add_to_cart':
        purchase.append(1 if np.random.rand() < 0.4 else 0)
    elif act == 'click':
        purchase.append(1 if np.random.rand() < 0.15 else 0)
    else:  # 'view'
        purchase.append(1 if np.random.rand() < 0.03 else 0)
purchase = np.array(purchase)

# Create the DataFrame
df = pd.DataFrame({
    'session_id': session_ids,
    'user_id': user_ids,
    'timestamp': timestamps,
    'action': actions,
    'price': prices,
    'product_category': categories,
    'purchase': purchase
})

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('ebay_data.csv', index=False)

print("\nDataset 'ebay_data.csv' generated successfully!")
