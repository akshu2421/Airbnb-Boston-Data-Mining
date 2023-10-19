#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 09:21:23 2023

@author: akshatathopte
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Load the dataset
listing_df = pd.read_csv('/Users/akshatathopte/Downloads/clean_listings.csv')  

# Create a new feature 'amenities_count'
listing_df['amenities_count'] = listing_df['amenities'].apply(lambda x: len(x.split(',')))

# Drop unnecessary columns
listing_df = listing_df[['amenities_count', 'property_type', 'neighbourhood_cleansed', 'price']]

# Remove rows with missing values
listing_df.dropna(subset=['amenities_count', 'property_type', 'neighbourhood_cleansed', 'price'], inplace=True)

# Convert price column to numeric
listing_df['price'] = listing_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Label encode categorical columns
label_encoder = LabelEncoder()
listing_df['property_type'] = label_encoder.fit_transform(listing_df['property_type'])
listing_df['neighbourhood_cleansed'] = label_encoder.fit_transform(listing_df['neighbourhood_cleansed'])

# Split the dataset into features and target variable
X = listing_df[['amenities_count', 'property_type', 'neighbourhood_cleansed']]
y = listing_df['price']

# Encode categorical variables using one-hot encoding
X = pd.get_dummies(X, columns=['property_type', 'neighbourhood_cleansed'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Ridge Regression model
regressor = Ridge()

# Define hyperparameters for grid search
param_grid = {'alpha': [0.1, 1, 10, 100],
              'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
grid_search = GridSearchCV(regressor, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Train Ridge Regression with best hyperparameters
regressor = Ridge(alpha=best_params['alpha'], solver=best_params['solver'])
regressor.fit(X_train, y_train)

# Make predictions on test set
y_pred = regressor.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Prompt the user to input the listing details
amenities_count = int(input("Enter the number of amenities in your listing: 1 - 14 "))
property_type = input("Enter the type of property (e.g. Apartment, House, Bed & Breakfast, Dorm, Boat, Entire floor, Camper/rv, Guest House, Loft, Condominium, Other, Villa or Townhouse): ")
neighbourhood_cleansed = input("Enter the neighbourhood in which your listing is located: Allston, Back Bay, Back Bay Village, Beacon Hill, Brighton, ChinaTown, CharlesTown, Dorchester, east Boston, Fenway, Hyde Park, Jamaica Plain, Leather Disrict, Longwood Medical Area, Mission Hill, Mattapan, Roxbury, North End, Roslindale, South Boston, South End, West End, West Roxbury or South Boston Waterfront  ")
