import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/train.csv')

df

df.dtypes

df.isnull().sum()

df.duplicated().sum()

df.describe()

df.replace({"NaN": np.nan}, regex=True, inplace = True)

df.isnull().sum()

df.drop('ID',axis=1,inplace=True)

df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min)','')
df=df.rename(columns={'Time_taken(min)':'Time_taken'})
df['Time_taken'] =df['Time_taken'].astype(int)

df['Weatherconditions']=df['Weatherconditions'].str.replace('conditions','')

from geopy.distance import geodesic

# Define a function to calculate distance
def calculate_distance(row):
    restaurant_coords = (row['Restaurant_latitude'], row['Restaurant_longitude'])
    delivery_coords = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
    return geodesic(restaurant_coords, delivery_coords).kilometers

# Apply the function to the DataFrame
df['distance_km'] = df.apply(calculate_distance, axis=1)
print(df)

df['Delivery_person_Age']=df['Delivery_person_Age'].astype(float)
df['Delivery_person_Ratings']=df['Delivery_person_Ratings'].astype(float)

df['Order_Date']=pd.to_datetime(df['Order_Date'])
df['Order_day']=df['Order_Date'].dt.day
df['Order_month']=df['Order_Date'].dt.month
df['Order_year']=df['Order_Date'].dt.year

col = df.pop("Order_day")
df.insert(10,"Order_day",col)

col = df.pop("Order_month")
df.insert(11,"Order_month",col)

col = df.pop("Order_year")
df.insert(12,"Order_year",col)

df['Time_Orderd']=pd.to_datetime(df['Time_Orderd'])

# Creating two new column for hour and minute
df['Hour_order']=df['Time_Orderd'].dt.hour
df['Min_order']=df['Time_Orderd'].dt.minute

col = df.pop("Hour_order")
df.insert(14,"Hour_order",col)

col = df.pop("Min_order")
df.insert(15,"Min_order",col)

df['Time_Order_picked']=pd.to_datetime(df['Time_Order_picked'])

# Creating two new column for hour and minute
df['Hour_order_picked']=df['Time_Order_picked'].dt.hour
df['Min_order_picked']=df['Time_Order_picked'].dt.minute

col = df.pop("Hour_order_picked")
df.insert(17,"Hour_order_picked",col)

col = df.pop("Min_order_picked")
df.insert(18,"Min_order_picked",col)

df['Delivery_person_Age']= df['Delivery_person_Age'].fillna(df.Delivery_person_Age.median())
df['Delivery_person_Ratings']=df['Delivery_person_Ratings'].fillna(df.Delivery_person_Ratings.median())
df['Time_Orderd']=df['Time_Orderd'].fillna(df.Time_Orderd.mode()[0])
df['Hour_order']=df['Hour_order'].fillna(df.Hour_order.mode()[0])
df['Min_order']=df['Min_order'].fillna(df.Min_order.mode()[0])
df['Weatherconditions']=df['Weatherconditions'].fillna(df.Weatherconditions.mode()[0])
df['Road_traffic_density']=df['Road_traffic_density'].fillna(df.Road_traffic_density.mode()[0])
df['multiple_deliveries']=df['multiple_deliveries'].fillna(df.multiple_deliveries.mode()[0])
df['Festival']=df['Festival'].fillna(df.Festival.mode()[0])
df['City']=df['City'].fillna(df.City.mode()[0])

df.isnull().sum()

plt.figure(figsize=(10,6))
sns.histplot(x=df.Delivery_person_Ratings,hue=df.City,bins=40,linewidth=2)

plt.figure(figsize=(10,6))
sns.barplot(x=df.Type_of_vehicle ,y=df.Time_taken,hue=df.Type_of_order)
plt.title('Time taken by vehicle according to order')

sns.scatterplot(x=df.Time_taken,y=df.distance_km)

sns.distplot(x=df.Delivery_person_Ratings)
plt.xlabel('Delivery person rating')

sns.scatterplot(x=df.Delivery_person_Age,y=df.Delivery_person_Ratings)

sns.scatterplot(x=df.Delivery_person_Age,y=df.Delivery_person_Ratings,hue=df.distance_km)

plt.figure(figsize=(8,5))
sns.countplot(x=df.City)

sns.distplot(x=df.Delivery_person_Age,kde=True,bins=30)

sns.barplot(x=df.Road_traffic_density,y=df.Time_taken)
plt.title('Time taken by road traffic density')

sns.countplot(x=df.Weatherconditions)

plt.figure(figsize=(10,4))
sns.countplot(x=df.Type_of_vehicle)
plt.title('Types of vehicle used for deliveries')

sns.countplot(x=df.multiple_deliveries,hue=df.City)

sns.countplot(x=df.Festival)

plt.figure(figsize=(10,4))
sns.barplot(x=df.Delivery_person_ID.value_counts().head(10).index,y=df.Delivery_person_ID.value_counts().head(10))
plt.xticks(rotation=50)
plt.title('Total No of delivery person id ')

x= df.drop(['Delivery_person_ID','Order_Date','Time_Orderd','Time_Order_picked','Time_taken'],axis=1)
y= df.Time_taken

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

x_train.shape,x_test.shape

y_train.shape,y_test.shape

x_train.select_dtypes(include='object').columns

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()

x_train[['Weatherconditions', 'Road_traffic_density',
       'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries', 'Festival',
       'City']] = oe.fit_transform(x_train[['Weatherconditions', 'Road_traffic_density',
       'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries', 'Festival',
       'City']])

x_test[['Weatherconditions', 'Road_traffic_density',
       'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries', 'Festival',
       'City']] = oe.transform(x_test[['Weatherconditions', 'Road_traffic_density',
       'Type_of_order', 'Type_of_vehicle', 'multiple_deliveries', 'Festival',
       'City']])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(x_train)

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()

lr.fit(x_train,y_train)
dtr.fit(x_train,y_train)
rfr.fit(x_train,y_train)

lr_pred = lr.predict(x_test)
dtr_pred = dtr.predict(x_test)
rfr_pred = rfr.predict(x_test)

from sklearn.metrics import r2_score

print(r2_score(y_test,lr_pred))
print(r2_score(y_test,dtr_pred))
print(r2_score(y_test,rfr_pred))

lr.score(x_train,y_train)

lr.score(x_test,y_test)

dtr.score(x_train,y_train)

dtr.score(x_test,y_test)

rfr.score(x_train,y_train)

rfr.score(x_test,y_test)

