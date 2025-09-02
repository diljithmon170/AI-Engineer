# import all the required libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# read the data
df1 = pd.read_csv("bengaluru_house_prices.csv")
# print(df1.head())

df1['area_type'].unique()
# print(df1['area_type'].value_counts())

# Drop features that are not required to build our model

df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
# print(df2.head())

# Data Cleaning: Handle NA values

# print(df2.isnull().sum())
df3 = df2.dropna()
# print(df3.isnull().sum())

# Feature Engineering
# Add new feature(integer) for bhk (Bedrooms Hall Kitchen)
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
# print(df3.bhk.unique())

# print(df3['total_sqft'].head(5))

# explore total_sqft feature
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# print(df3[~df3['total_sqft'].apply(is_float)].head(10))

"""Above shows that total_sqft can be a range (e.g. 2100-2850). For such case
 we can just take average of min and max value in the range. There are other 
 cases such as 34.46Sq. Meter which one can convert to square ft using unit 
 conversion. I am going to just drop such corner cases to keep things simple """

def convert_sqft_to_num(x):
    try:
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
# print(df4.head(3))

# Feature Engineering - Add new feature called price per square feet

df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
# print(df5.head())

# Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
# print(location_stats)

# Dimensionality Reduction
"""Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns"""

location_stats_less_than_10 = location_stats[location_stats <= 10]
# print(location_stats_less_than_10)

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# print(df5.location.value_counts())

# Outlier Removal Using Business Logic
df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
# print(df6.shape)

# Outlier Removal Using Standard Deviation and Mean
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
# print(df7.shape)

# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()
# plot_scatter_chart(df7, "Rajaji Nagar")
# plot_scatter_chart(df7, "Hebbal")

# Remove properties where price of 2 BHK is more than 3 BHK (at same location)
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values
                )
    return df.drop(exclude_indices, axis='index')
df8 = remove_bhk_outliers(df7)
# print(df8.shape)

# Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties
# plot_scatter_chart(df8, "Rajaji Nagar")

df9 =df8[df8.bath < df8.bhk + 2]
# print(df9.shape)

# print(df9.head(3))

# remove columns that are not required for model building
df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
print(df10.head(3))