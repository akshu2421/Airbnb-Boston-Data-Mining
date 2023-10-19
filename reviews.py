

# Loading libraries 
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
import re
import numpy as np

# Import the dataset reviews.csv
reviews_data = pd.read_csv("/Users/akshatathopte/Documents/Airbnb_project_data mining/reviews.csv")
reviews_data = reviews_data.dropna()
reviews_data.head()

# Check column
reviews_data.columns

#
num_rows = reviews_data.shape[0]  # Number of rows
num_cols = reviews_data.shape[1]  # Number of columns

print("Number of rows:", num_rows)
print("Number of columns:", num_cols)


# Polarity score to each comment positive, negative and neutral
from nltk.sentiment.vader import SentimentIntensityAnalyzer
senti = SentimentIntensityAnalyzer()

reviews_data['polarity_value'] = "Default"
reviews_data['neg'] = 0.0
reviews_data['pos'] = 0.0
reviews_data['neu'] = 0.0
reviews_data['compound'] = 0.0

for index, row in reviews_data.iterrows():
    ss = senti.polarity_scores(row['comments'])
    reviews_data.at[index, 'polarity_value'] = ss
    reviews_data.at[index, 'neg'] = ss['neg']
    reviews_data.at[index, 'pos'] = ss['pos']
    reviews_data.at[index, 'neu'] = ss['neu']
    reviews_data.at[index, 'compound'] = ss['compound']

reviews_data.head()

# Save to csv
reviews_data.to_csv('polarity_reviews.csv')

# Remove language that is not in english
from langdetect import detect

def detect_lang(sente):
    sente = str(sente)
    try:
        return detect(sente)
    except:
        return "None"

for index, row in reviews_data.iterrows():
    lang = detect_lang(row['comments'])
    reviews_data.at[index, 'language'] = lang


# Take rows whose language is English
eng_reviews_data = reviews_data[reviews_data.language == 'en']

eng_reviews_data.head(10)

eng_reviews_data.columns

# Save to csv
reviews_data.to_csv('cleaned_data_reviews.csv')

# Visualize polarity score for Positivity
pos_polar = eng_reviews_data[['pos']]
pos_polar = pos_polar.groupby(pd.cut(pos_polar["pos"], np.arange(0, 1.1, 0.1))).count()
pos_polar = pos_polar.rename(columns={'pos': 'count_of_Comments'})
pos_polar = pos_polar.reset_index()
pos_polar = pos_polar.rename(columns={'pos': 'range_i'})

for i, r in pos_polar.iterrows():
    pos_polar.at[i, 'RANGE'] = float(str(r['range_i'])[1:4].replace(',', ''))
    pos_polar.at[i, 'Sentiment'] = 'positive'
del pos_polar['range_i']
pos_polar.head()

# Output
#   count_of_Comments  RANGE Sentiment
#0               2018    0.0  positive
#1              12482    0.1  positive
#2              26385    0.2  positive
#3              25510    0.3  positive
#4              16222    0.4  positive

# Visualize polarity score for negativity
neg_polar = eng_reviews_data[['neg']]
neg_polar = neg_polar.groupby(pd.cut(neg_polar["neg"], np.arange(0, 1.1, 0.1))).count()
neg_polar = neg_polar.rename(columns={'neg': 'count_of_Comments'})
neg_polar = neg_polar.reset_index()
neg_polar = neg_polar.rename(columns={'neg': 'range_i'})

for i, r in neg_polar.iterrows():
    neg_polar.at[i, 'RANGE'] = float(str(r['range_i'])[1:4].replace(',', ''))
    neg_polar.at[i, 'Sentiment'] = 'negative'
del neg_polar['range_i']

for i, r in neg_polar.iterrows():
    neg_polar = neg_polar.append(pd.Series([r[0], r[1], r[2]], index=['count_of_Comments', 'RANGE', 'Sentiment']),
                                   ignore_index=True)
    
neg_polar.head()


# Output
#   count_of_Comments  RANGE Sentiment
#0              18017    0.0  negative
#1               1691    0.1  negative
#2                217    0.2  negative
#3                 41    0.3  negative
#4                 24    0.4  negative


# Visualize polarity score for neutrality
neu_polar = eng_reviews_data[['neu']]
neu_polar = neu_polar.groupby(pd.cut(neu_polar["neu"], np.arange(0, 1.0, 0.1))).count()
neu_polar = neu_polar.rename(columns={'neu': 'count_of_Comments'})
neu_polar = neu_polar.reset_index()
neu_polar = neu_polar.rename(columns={'neu': 'range_i'})

for i, r in neu_polar.iterrows():
    neu_polar.at[i, 'RANGE'] = float(str(r['range_i'])[1:4].replace(',', ''))
    neu_polar.at[i, 'Sentiment'] = 'neutrl'
del neu_polar['range_i']

for i, r in neu_polar.iterrows():
    neu_polar = neg_polar.append(pd.Series([r[0], r[1], r[2]], index=['count_of_Comments', 'RANGE', 'Sentiment']),
                                ignore_index=True)

neu_polar.head()

# Output
#   count_of_Comments  RANGE Sentiment
#0              18017    0.0  negative
#1               1691    0.1  negative
#2                217    0.2  negative
#3                 41    0.3  negative
#4                 24    0.4  negative

# Positive comments

plt.figure(figsize=(10,10))
sns.factorplot(data=pos_polar, x="RANGE", y="count_of_Comments",col="Sentiment", color='blue') 
plt.title("Positive Comments")  
plt.show()  

# Negative Comments

plt.figure(figsize=(10,10))
sns.factorplot(data=neg_polar, x="RANGE", y="count_of_Comments",col="Sentiment", color='blue') 
plt.title("Negative Comments")  
plt.show() 

# Neutral Comments

plt.figure(figsize=(10,10))
sns.factorplot(data=neu_polar, x="RANGE", y="count_of_Comments",col="Sentiment", color='blue') 
plt.title("Neutral Comments")  
plt.show() 

# The analysis reveals that the majority of the comments are classified as neutral, with almost no texts being classified as significantly negative. 
# Many comments have a negativity score of exactly 0.0. However, a significant portion of the comments are classified as positive. 
# The number of reviews can be loosely interpreted as the number of times people have stayed in the listed property, depending on factors such as the listing's visibility and duration of appearance.

# Import Dataset listing.csv
listing_data = pd.read_csv('/Users/akshatathopte/Documents/Airbnb_project_data mining/listings (2).csv')
listing_data=listing_data[['number_of_reviews','price','review_scores_rating']]

# Data Cleaning 

# Checking missing values in data
listing_data.isnull().sum()

# Replac NaN values with 0
listing_data.fillna(0, inplace=True)

# Pulling prices from the table
price = listing_data['price']
prices=[]

# Converting data to float
for p in price:
    p=float(p[1:].replace(',',''))
    prices.append(p)

# Replace the price column with the new column
listing_data['price']=prices
price_review = listing_data[['number_of_reviews', 'price']].sort_values(by = 'price')

price_review.plot(x = 'price', 
                  y = 'number_of_reviews', 
                  style = 'o',
                  figsize =(12,8),
                  legend = False,
                  title = 'Reviews based on Price')

plt.xlabel("price")
plt.ylabel("Number of reviews")




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Import Dataset listing.csv
listing_data = pd.read_csv('/Users/akshatathopte/Documents/Airbnb_project_data mining/listings (2).csv')
listing_data = listing_data[['number_of_reviews', 'price', 'review_scores_rating']]

# Data Cleaning

# Checking missing values in data
listing_data.isnull().sum()

# Replace NaN values with 0
listing_data.fillna(0, inplace=True)

# Pulling prices from the table
price = listing_data['price']
prices = []

# Converting data to float
for p in price:
    p = float(p[1:].replace(',', ''))
    prices.append(p)

# Replace the price column with the new column
listing_data['price'] = prices

# Sort the data by price
price_review = listing_data[['number_of_reviews', 'price']].sort_values(by='price')

# Plot the data points
plt.figure(figsize=(12, 8))
plt.plot(price_review['price'], price_review['number_of_reviews'], 'o', color ='blue')
plt.xlabel("Price")
plt.ylabel("Number of Reviews")
plt.title("Reviews based on Price")

# Fit linear regression to the data
X = price_review[['price']]
y = price_review['number_of_reviews']
reg = LinearRegression().fit(X, y)
plt.plot(X, reg.predict(X), color='red', linewidth=2)

plt.show()


