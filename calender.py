

##loading libraries 
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

# Import the Dataset Calender.csv
calender_data = pd.read_csv("/Users/akshatathopte/Documents/Airbnb_project_data mining/calendar.csv")

# Check column
calender_data.columns

num_rows = calender_data.shape[0]  # Number of rows
num_cols = calender_data.shape[1]  # Number of columns

print("Number of rows:", num_rows)
print("Number of columns:", num_cols)


### Data cleaning and preprocessing

# Checking missing values in data
calender_data.isnull().sum()

# Minimum_nights and Maximum nights column has null values

# Replacing NaN values with 0
calender_data.fillna(0, inplace=True)
calender_data = calender_data[calender_data.price != 0]

# Pulling prices from the table
price = calender_data['price']
prices = []

for p in price:
    p = str(p)  # Convert to string
    p = re.sub('[^0-9.]+', '', p)
    prices.append(float(p))
    
# Replacing the price column with the new column
calender_data['price'] = prices
calender_data = calender_data[calender_data.price >= 0]

# Splitting date column into day month and year
calender_data['Year'],calender_data['Month'],calender_data['Day']=calender_data['date'].str.split('-',2).str
calender_data.head()


# Group and calculate mean
year_df = calender_data.groupby(['Year', 'Month']).agg({'price': 'mean', 'maximum_nights': 'first', 'minimum_nights': 'first'}).reset_index()
year_df=year_df.reset_index()

# Create average price  column
year_df=year_df.rename(columns={'price':'average_Price'})
year_df['year-Month']=year_df['Year'].map(str) + "-" + year_df['Month'].map(str)

year_df.columns

# Save to CSV
year_df.to_csv('calender_data_cleaned.csv')

# Grouping and calculating mean
year_df = year_df.groupby(['Year', 'Month'])['average_Price'].mean()
year_df = year_df.reset_index()

# Creating year-Month column
year_df['year-Month'] = year_df['Year'].map(str) + "-" + year_df['Month'].map(str)

# Save to CSV
year_df.to_csv('year_month_data.csv')

# Printing the first few rows
year_df.head()

# Output
#   Year Month  average_Price year-Month
#0  2023    03     190.275775    2023-03
#1  2023    04     227.551186    2023-04
#2  2023    05     249.852830    2023-05
#3  2023    06     258.786953    2023-06
#4  2023    07     260.886466    2023-07


# It can be seen that the data is available of year 2023 and when average prices are analyzed maximum rates for the listings were in the month of september with 260.88 as average price. Visualizing the same for a better understanding

# visualizing the trend of year/Month and average prices of the listing

objects = year_df['year-Month']
y_pos = year_df['average_Price']

fig, ax = plt.subplots(figsize =(15,8))

year_df.plot(kind='barh', 
           y='average_Price',  # Use 'average_Price' as the y-axis
           x='year-Month',  # Use 'year-Month' as the x-axis
           color = 'grey', 
           ax=ax)  # Pass ax to specify the Axes object

ax.set_title('Boston Airbnb prices with Months and Year')  # Update the title
ax.set_xlabel('Average Price')  # Update the label for the x-axis
ax.set_ylabel('Year-Month')  # Update the label for the y-axis

# Add labels to the bars with one decimal point
for i, v in enumerate(y_pos):
    ax.text(v + 0.1, i, f'{v:.1f}', color='black')

plt.show()

# Output 
# It can be clearly seen that the maximum average price for listings are from month of september and october 2023. The month from september and october are growing and the reason is because of good weather and Massachussetts' best time to observe fall colors.Fall Colors in Massachusetts attracts a lot of visitors which makes September and October peak months for Airbnb hosts. But in the month of november december it is declining to know reason we will explore further
# Before getting into month november december lets firts analyze day from given date and checked weather it was a holiday and what is the reason for that holiday.


#Checking day name from date data and holidays
from datetime import date
import datetime
import calendar
import holidays

calender_data.fillna(0, inplace=True)
us_holidays = holidays.US()

calender_data['day_Name'] = 'default'
calender_data['holiday'] = 'False'
calender_data['us_holidays_name'] = 'working'

for index, row in calender_data.iterrows():
    sdate = datetime.date(int(row['Year']), int(row['Month']), int(row['Day']))
    vall = date(int(row['Year']), int(row['Month']), int(row['Day'])) in us_holidays
    calender_data.at[index, 'day_Name'] = calendar.day_name[sdate.weekday()]
    calender_data.at[index, 'holiday'] = vall
    calender_data.at[index, 'us_holidays_name'] = us_holidays.get(sdate)

calender_data.to_csv('holidays_data.csv')
calender_data.head()

# Added 3 new columns Day_Name, Holiday and us_holiday_name

# Calculating Average price for each day
day_df = calender_data.groupby('day_Name').price.mean()
day_df = day_df.reset_index()
day_df['day_num'] = 0

for index, row in day_df.iterrows():
    if row['day_Name'] == 'Monday':
        day_df.at[index, 'day_num'] = 1
    if row['day_Name'] == 'Tuesday':
        day_df.at[index, 'day_num'] = 2
    if row['day_Name'] == 'Wednesday':
        day_df.at[index, 'day_num'] = 3
    if row['day_Name'] == 'Thursday':
        day_df.at[index, 'day_num'] = 4
    if row['day_Name'] == 'Friday':
        day_df.at[index, 'day_num'] = 5
    if row['day_Name'] == 'Saturday':
        day_df.at[index, 'day_num'] = 6
    if row['day_Name'] == 'Sunday':
        day_df.at[index, 'day_num'] = 7

day_df = day_df.sort_values('day_num', ascending=[1])
day_df = day_df.rename(columns={'price': 'Average_Price'})
day_df

# Output
#    day_Name  Average_Price  day_num
#1     Monday     235.970198        1
#5    Tuesday     234.583016        2
#6  Wednesday     235.687100        3
#4   Thursday     241.747615        4
#0     Friday     258.210598        5
#2   Saturday     260.488087        6
#3     Sunday     240.933981        7

# It can be seen that the average price of listings increases on weekends i.e on saturday is most of average price with 260.488

#checking which holiday has maximum listings
holiday_df=calender_data.groupby('us_holidays_name').listing_id.count()
holiday_df=holiday_df.reset_index()
holiday_df=holiday_df.sort_values('listing_id',ascending=[0])
holiday_df.head(20)

#                        us_holidays_name  listing_id
#0                          Christmas Day        3863
#1                           Columbus Day        3863
#2                       Independence Day        3863
#3   Juneteenth National Independence Day        3863
#4                              Labor Day        3863
#5             Martin Luther King Jr. Day        3863
#6                           Memorial Day        3863
#7                         New Year's Day        3863
#8                           Thanksgiving        3863
#9                           Veterans Day        3863
#10               Veterans Day (Observed)        3863
#11                 Washington's Birthday        3863


## 11 us holidays have highest listing 


# find which holiday has the maximum average price.
holiday_price_df=calender_data.groupby('us_holidays_name').price.mean()
holiday_price_df=holiday_price_df.reset_index()
holiday_price_df=holiday_price_df.sort_values('price',ascending=[0])

holiday_price_df.head(10)

# Output
#                        us_holidays_name       price
#2                       Independence Day  251.124256
#1                           Columbus Day  250.106135
#3   Juneteenth National Independence Day  247.557857
#9                           Veterans Day  247.304685
#10               Veterans Day (Observed)  246.581931
#7                         New Year's Day  245.706705
#4                              Labor Day  243.778152
#11                 Washington's Birthday  236.602899
#6                           Memorial Day  236.341962
#8                           Thanksgiving  232.817240

# visualizing the same
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8)) 
ax = sns.barplot(y="us_holidays_name", x="price", data=holiday_price_df, color='royalblue')
ax.set(xlabel='Average Price', ylabel='US Holidays Name') # Update x and y labels
plt.title('Average Price by US Holidays Name') # Add a title to the plot
plt.show()

#Independence day has highest price

# Merge the df holiday_df and holiday_price_df
merge_df=pd.merge(holiday_df,holiday_price_df,on='us_holidays_name')
merge_df=merge_df.rename(columns={'listing_id':'number_Of_Listings'})
merge_df=merge_df.rename(columns={'price':'average_Price'})
merge_df


#analyzing longweekednd holiday days
# prices increase on long weeknd

# analyzing data from date 1st of july to date 10th of july which includes both long weekend and normal workdays

july_month = calender_data[(calender_data['Year'] == '2023') & (calender_data['Month'] == '06') & ((calender_data['Day'].astype(int) >= 1) & (calender_data['Day'].astype(int) <= 10))]

# average price
july_month = july_month.groupby('Day').price.mean()
july_month = july_month.reset_index()
july_month = july_month.sort_values('Day', ascending=[1])
july_month = july_month.rename(columns={'price': 'Average_Price'})
july_month.head(10)

# Output
#  Day  Average_Price
#0  01     259.009578
#1  02     281.979550
#2  03     277.807662
#3  04     251.736992
#4  05     256.191043
#5  06     257.017085
#6  07     257.621279
#7  08     254.753559
#8  09     262.618949
#9  10     264.259902

# Line Graph for same
import numpy as np
import matplotlib.pyplot as plt

x = july_month['Day'].tolist() # Get the days as x values
y = july_month['Average_Price'].tolist() # Get the average prices as y values

plt.plot(x, y, 'bo-') # Update color to blue
plt.ylabel('Average Price')
plt.xlabel('Day')
plt.xticks(ticks=x, labels=x) 
plt.title('Day with Average Price')
plt.show()

# day 1 and 2 is sat and sun due to which we have high price its dropping on 4th july being an holiday price drops as day is tuesday





