
'''WHAT ARE THE FACTORS THAT INFLUENCE THE PRICES OF LISTING'''

'''ANALYSIS-1'''
# Cleaning the data

# Import the libraries
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

# Import the Dataset listing.csv
DF = pd.read_csv('listings.csv')
DF=DF[['id','name','summary','longitude','latitude','space','description','instant_bookable','neighborhood_overview','neighbourhood_cleansed','host_id','host_name','host_since', 'host_response_time','street', 'zipcode','review_scores_rating','property_type','room_type','accommodates','bathrooms','bedrooms','beds','reviews_per_month','amenities','cancellation_policy','number_of_reviews','price']]

num_rows = DF.shape[0]  # Number of rows
num_cols = DF.shape[1]  # Number of columns

print("Number of rows:", num_rows)
print("Number of columns:", num_cols)

# Replacing NaN to 0
DF.fillna(0, inplace=True)

# Extracting prices 
price = DF['price']
prices=[]

# Clear the data in order for it to float
for p in price:
    p=float(p[1:].replace(',',''))
    prices.append(p)

# Replace the price column to new column
DF['price']=prices

# Excluding listings with 0 for price, beds, bedrooms, and accommodations, etc.
DF = DF[DF.bathrooms >0]
DF = DF[DF.bedrooms > 0] 
DF = DF[DF.beds > 0] 
DF = DF[DF.price  > 0]
DF = DF[DF.review_scores_rating  > 0]
DF = DF[DF.reviews_per_month > 0]
DF = DF[DF.accommodates  > 0]

DF.head()

#O/P: 
#    id  ... price
#1   3075044  ...  65.0
#2      6976  ...  65.0
#3   1436513  ...  75.0
#4   7651065  ...  79.0
#5  12386020  ...  75.0

# Categorizing differernt listings based on room_type
roomType_DF = (DF.groupby('room_type')
                      .agg(number_Of_Listings=('id', 'count'))
                      .reset_index())
print(roomType_DF)

#O/P 
 #room_type  number_Of_Listings
#0  Entire home/apt                1393
#1     Private room                1061
#2      Shared room                  52

# The amount of postings depending on room type is shown above. Visualizing the same thing will give  additional clarity.


# Create a sample DataFrame for illustration purposes 
#horizontal bar graph 

# Analyzing and visualizing the amount of listings by property type

import seaborn as sns
import matplotlib.pyplot as plt

# Get the number of listings per property type
propertytype_count = DF['property_type'].value_counts()

plt.figure(figsize=(15, 8))
ax = sns.countplot(data=DF, y='property_type', color='#66c2ff')

# Add the number of listings to each bar
for i, count in enumerate(propertytype_count):
    ax.annotate(str(count), xy=(count+10, i), va='center')

plt.title('Boston Property Type Frequency')
plt.xlabel('Number of Listings')
plt.ylabel('Property Type')
plt.show()


print(DF.columns)


# Analyzing the prices for different room type and property type
roomProperty_DF = DF.groupby(['property_type','room_type']).price.mean()
roomProperty_DF = roomProperty_DF.reset_index()
roomProperty_DF=roomProperty_DF.sort_values('price',ascending=[0])
roomProperty_DF.head()
'''
 property_type        room_type       price
22     Townhouse  Entire home/apt  320.800000
13    Guesthouse  Entire home/apt  289.000000
7           Boat     Private room  287.000000
14         House  Entire home/apt  286.809917
6           Boat  Entire home/apt  275.222222
'''
# Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# Group the DataFrame by property type and room type and calculate the mean price
grouped_DF = DF.groupby(['property_type', 'room_type']).price.mean().unstack()

# Plot the heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(grouped_DF, annot=True, fmt=".0f", cmap='Blues')
plt.title('Mean Price by Property Type and Room Type')
plt.xlabel('Room Type')
plt.ylabel('Property Type')
plt.show()

# Heatmap indicating listing price fluctuation based on number of bedrooms
plt.figure(figsize=(12,12))
sns.heatmap(DF.groupby([
        'neighbourhood_cleansed', 'bedrooms']).price.mean().unstack(), annot=True, fmt=".0f", cmap="YlGnBu")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Neighbourhood Cleansed")
plt.title("Average Price by Neighbourhood and Number of Bedrooms")
plt.show()


#Analyzing what amenities are most common 
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Define a function to clean and tokenize the amenities column
def clean_amenities(amenities):
    p = re.sub('[^a-zA-Z]+',' ', amenities)
    tokens = nltk.word_tokenize(p.lower())
    lemmatizer = nltk.WordNetLemmatizer()
    filtered = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    return filtered

# Get the top rows by frequency of amenities
amenities_freq = DF['amenities'].apply(clean_amenities).explode().value_counts().nlargest(100)

# Create a word cloud from the most common amenities
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(amenities_freq)

# Display the word cloud
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

DF.to_csv('clean_listings.csv', index=False)
