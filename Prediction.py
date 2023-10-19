#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 03:31:49 2023

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


# Load the dataset
listing_df = pd.read_csv('/Users/akshatathopte/Downloads/clean_listings.csv')  
listing_df.columns

# Select only columns with numeric values
numeric_cols = listing_df.select_dtypes(include=np.number)

# Calculate the mean
mean = numeric_cols.mean().mean()

# Calculate the median
median = numeric_cols.median().median()

# Print the results
print("Mean:", mean)
print("Median:", median)

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

# Create and train the Multiple Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prompt the user to input the listing details
amenities_count = int(input("Enter the number of amenities in your listing: 1 - 14 "))
property_type = input("Enter the type of property (e.g. Apartment, House, Bed & Breakfast, Dorm, Boat, Entire floor, Camper/rv, Guest House, Loft, Condominium, Other, Villa or Townhouse): ")
neighbourhood_cleansed = input("Enter the neighbourhood in which your listing is located: Allston, Back Bay, Back Bay Village, Beacon Hill, Brighton, ChinaTown, CharlesTown, Dorchester, east Boston, Fenway, Hyde Park, Jamaica Plain, Leather Disrict, Longwood Medical Area, Mission Hill, Mattapan, Roxbury, North End, Roslindale, South Boston, South End, West End, West Roxbury or South Boston Waterfront  ")




