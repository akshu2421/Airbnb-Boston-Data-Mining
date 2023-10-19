#ANALYSIS - 2
#Where in BOSTON Should You Invest in Real Estate to Get the Best Airbnb Returns?It has already been determined that the greatest number of listings are for Entire Home/Apartment.  average costs for these postings depending on the room type.

# import libraies and cleaning the data

import pandas as pd
import numpy as np

listing_df = pd.read_csv('/Users/rakeshreddy/Downloads/listings.csv')

listing_df= listing_df[['id','name','summary','longitude','latitude','space','description','instant_bookable','neighborhood_overview','neighbourhood_cleansed','host_id','host_name','host_since',
                 'host_response_time','street', 'zipcode','review_scores_rating','property_type','room_type','accommodates','bathrooms','bedrooms','beds','reviews_per_month','amenities','cancellation_policy','number_of_reviews','price']]

# replacing NaN = 0

listing_df.fillna(0, inplace=True)

#Extracting prices 
price = listing_df['price']
prices=[]

#clear the data in order for it to float
for p in price:
    p=float(p[1:].replace(',',''))
    prices.append(p)

#replace the price column to new column
listing_df['price']=prices

#excluding listings with 0 for price, beds, bedrooms, and accommodations, etc.
listing_df = listing_df[listing_df.bathrooms >0]
listing_df = listing_df[listing_df.bedrooms > 0]
listing_df = listing_df[listing_df.beds > 0]
listing_df = listing_df[listing_df.price  > 0]
listing_df = listing_df[listing_df.review_scores_rating  > 0]
listing_df = listing_df[listing_df.reviews_per_month > 0]
listing_df = listing_df[listing_df.accommodates  > 0]

# Average listing price for each item type

avgPrice_DF=listing_df.groupby('room_type').price.mean()
avgPrice_DF=avgPrice_DF.reset_index()
avgPrice_DF=avgPrice_DF.rename(columns={'price':'average_Price'})
avgPrice_DF

# Geographical Clusters to determine which region in Boston has the most Airbnb listings.
# categorizing each sort of property

home = listing_df[(listing_df.room_type == 'Entire home/apt')]
private = listing_df[(listing_df.room_type == 'Private room')]
shared = listing_df[(listing_df.room_type == 'Shared room')]

location_home = home[['latitude', 'longitude']]
location_private = private[['latitude', 'longitude']]
location_shared = shared[['latitude', 'longitude']]

# Geographical Clusters to determine which region in Boston has the most Airbnb listings.
# categorizing each sort of property

home = listing_df[(listing_df.room_type == 'Entire home/apt')]
private = listing_df[(listing_df.room_type == 'Private room')]
shared = listing_df[(listing_df.room_type == 'Shared room')]

location_home = home[['latitude', 'longitude']]
location_private = private[['latitude', 'longitude']]
location_shared = shared[['latitude', 'longitude']]



