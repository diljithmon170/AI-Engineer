import pandas as pd
mydata = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
}
a=[1, 2, 3, 4, 5]

df = pd.Series(a, index = ["a", "b", "c", "d", "e"])
print(df["b"])

print(pd.__version__)

# key/value object

calories = {"day1": 420, "day2": 380, "day3": 390}
myvar = pd.Series(calories)
print(myvar)

# dataframe from dictionary

data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

myvar = pd.DataFrame(data, index=["day1", "day2", "day3"])
print(myvar)  # Accessing the first row
pd.options.display.max_rows = 99
df1 = pd.read_csv("data.csv")
print(df1)