import pandas as pd
df = pd.read_csv("carprices.csv")
# print(df)

# create dummy variables as 0 and 1
dummies = pd.get_dummies(df.CarModel).astype(int)
# print(dummies)

# concat the two dataframes
merged = pd.concat([df, dummies], axis='columns')
# print(merged)

# drop the CarModel column and one dummy variable column to avoid the dummy variable trap
final = merged.drop(['CarModel', 'BMW X5'], axis='columns')
# print(final)


# define x and y variables for the model
x=final.drop('SellPrice($)', axis='columns')
y=final['SellPrice($)']
# print(x)
# print(y)

# create and train the model 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)

# predict the price of a car with given features
# print(model.predict(x))

#Predict price of a mercedez benz that is 4 yr old with mileage 45000
print(model.predict([[45000, 4, 0, 1]]))

#Predict price of a BMW X5 that is 7 yr old with mileage 86000
print(model.predict([[86000, 7, 0, 0]]))

# model score
print(model.score(x,y))