import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Excel file
data = pd.read_excel('/Users/limitlesslegacy/Downloads/Data_Train.xlsx')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display information about the dataset
print("\nInformation about the dataset:")
print(data.info())

# Display the shape of the dataset
print("\nShape of the dataset:")
print(data.shape)

# Display the count of non-null values in each column
print("\nCount of non-null values in each column:")
print(data.count())

# Display the data types of each column
print("\nData types of each column:")
print(data.dtypes)

# Display basic statistical information about the dataset
print("\nStatistical information about the dataset:")
print(data.describe())

# Display the count of missing values in each column
print("\nCount of missing values in each column:")
print(data.isna().sum())

# Display rows where 'Route' or 'Total_Stops' is null
print("\nRows with missing 'Route' or 'Total_Stops':")
print(data[data['Route'].isna() | data['Total_Stops'].isna()])

# Drop rows with missing values
data.dropna(inplace=True)

# Display the count of missing values after dropping them
print("\nCount of missing values after dropping:")
print(data.isna().sum())

# Display the count of non-null values after dropping missing values
print("\nCount of non-null values after dropping:")
print(data.count())

"""LET'S GET STARTED !!!"""

# Function to convert duration to minutes
def convert_duration(duration):
    if len(duration.split()) == 2:
        hours = int(duration.split()[0][:-1])
        minutes = int(duration.split()[1][:-1])
        return hours * 60 + minutes
    else:
        return int(duration[:-1]) * 60

# Apply the convert_duration function to the 'Duration' column
data['Duration'] = data['Duration'].apply(convert_duration)

# Display the first few rows after converting duration
print("\nFirst few rows after converting duration:")
print(data.head())

# Convert 'Dep_Time' and 'Arrival_Time' columns to datetime format
data['Dep_Time'] = pd.to_datetime(data['Dep_Time'])
data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'])

# Display the data types after converting to datetime
print("\nData types after converting to datetime:")
print(data.dtypes)

# Extract hour and minute from 'Dep_Time' and 'Arrival_Time' columns
data['Dep_Time_in_hours'] = data['Dep_Time'].dt.hour
data['Dep_Time_in_minutes'] = data['Dep_Time'].dt.minute
data['Arrival_Time_in_hours'] = data['Arrival_Time'].dt.hour
data['Arrival_Time_in_minutes'] = data['Arrival_Time'].dt.minute

# Display the first few rows after extracting time information
print("\nFirst few rows after extracting time information:")
print(data.head())

# Drop 'Dep_Time' and 'Arrival_Time' columns
data.drop(['Dep_Time', 'Arrival_Time'], axis=1, inplace=True)

# Display the first few rows after dropping 'Dep_Time' and 'Arrival_Time'
print("\nFirst few rows after dropping 'Dep_Time' and 'Arrival_Time':")
print(data.head())

# Convert 'Date_of_Journey' column to datetime format
data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], dayfirst=True)

# Display the first few rows after converting 'Date_of_Journey'
print("\nFirst few rows after converting 'Date_of_Journey':")
print(data.head())

# Display unique years in 'Date_of_Journey' column
print("\nUnique years in 'Date_of_Journey' column:")
print(data['Date_of_Journey'].dt.year.unique())

# Extract day and month from 'Date_of_Journey' column
data['Day'] = data['Date_of_Journey'].dt.day
data['Month'] = data['Date_of_Journey'].dt.month

# Display the first few rows after extracting day and month
print("\nFirst few rows after extracting day and month:")
print(data.head())

# Drop 'Date_of_Journey' column
data.drop('Date_of_Journey', axis=1, inplace=True)

# Display the first few rows after dropping 'Date_of_Journey'
print("\nFirst few rows after dropping 'Date_of_Journey':")
print(data.head())

# Display the value counts of 'Total_Stops' column
print("\nValue counts of 'Total_Stops' column:")
print(data['Total_Stops'].value_counts())

# Map categorical values in 'Total_Stops' to numerical values
data['Total_Stops'] = data['Total_Stops'].map({
    'non-stop': 0,
    '1 stop': 1,
    '2 stop': 2,
    '3 stop': 3,
    '4 stop': 4,
})

# Display the first few rows after mapping 'Total_Stops'
print("\nFirst few rows after mapping 'Total_Stops':")
print(data.head())

# Display the value counts of 'Additional_Info' column
print("\nValue counts of 'Additional_Info' column:")
print(data['Additional_Info'].value_counts())

# Drop 'Additional_Info' column
data.drop('Additional_Info', axis=1, inplace=True)

# Display the first few rows after dropping 'Additional_Info'
print("\nFirst few rows after dropping 'Additional_Info':")
print(data.head())

# Display the columns with object data type
print("\nColumns with object data type:")
print(data.select_dtypes(['object']).columns)

# Plot countplots for categorical columns
for i in ['Airline', 'Source', 'Destination', 'Route']:
    plt.figure(figsize=(15, 6))
    sns.countplot(data=data, x=i)
    ax = sns.countplot(x=i, data=data.sort_values('Price', ascending=True))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    plt.tight_layout()
    plt.show()
    print('\n\n')

# Display the value counts of 'Airline' column
print("\nValue counts of 'Airline' column:")
print(data['Airline'].value_counts())

# Plot barplot for 'Airline' vs 'Price'
plt.figure(figsize=(15, 6))
ax = sns.barplot(x='Airline', y='Price', data=data.sort_values('Price', ascending=False))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.tight_layout()
plt.show()

# Plot boxplot for 'Airline' vs 'Price'
plt.figure(figsize=(15, 6))
ax = sns.boxplot(x='Airline', y='Price', data=data.sort_values('Price', ascending=False))
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.tight_layout()
plt.show()

# Display statistics of 'Price' grouped by 'Airline'
print("\nPrice statistics grouped by Airline:")
print(data.groupby('Airline').describe()['Price'].sort_values('mean', ascending=False))

'''ENCODING :'''
# Encoding categorical variable 'Airline'
Airline = pd.get_dummies(data['Airline'], drop_first=True)
print("\nEncoded 'Airline' column:")
print(Airline.head())

# Concatenate the encoded 'Airline' column with the original dataset
data = pd.concat([data, Airline], axis=1)
print("\nDataset after concatenating encoded 'Airline' column:")
print(data.head())

# Drop the original 'Airline' column
data.drop('Airline', axis=1, inplace=True)
print("\nDataset after dropping original 'Airline' column:")
print(data.head())

#SOURCE AND DESTINATION
list1 = ['Source', 'Destination']
for i in list1:
    print(data[i].value_counts(), '\n')
    
'''Encoding'''
data = pd.get_dummies(data=data,columns=list1,drop_first=True)
print(data.head())



#ROUTE
from sklearn.preprocessing import LabelEncoder

# Create a new DataFrame with the 'Route' column
route = data[['Route']].copy()

# Split the 'Route' column into separate columns
split_routes = route['Route'].str.split('→', expand=True)

# Assign the split values to new columns in the 'Route' DataFrame
route[['Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5']] = split_routes.iloc[:, :5]

# Display the first few rows of the 'Route' DataFrame after splitting
print(route.head())

# Fill missing values with 'None' in the 'Route' DataFrame
route.fillna('None', inplace=True)

# Display the first few rows of the 'Route' DataFrame after filling missing values
print(route.head())

# Apply LabelEncoder to each 'Route' column
for i in range(1, 6):
    col = 'Route_' + str(i)
    le = LabelEncoder()
    route[col] = le.fit_transform(route[col])

# Drop the original 'Route' column
route.drop('Route', axis=1, inplace=True)

# Display the first few rows of the 'Route' DataFrame after LabelEncoding
print(route.head())

# Concatenate 'route' DataFrame with the original 'data' DataFrame
data = pd.concat([data, route], axis=1)

# Drop the original 'Route' column from the 'data' DataFrame
data.drop('Route', axis=1, inplace=True)

# Display the first few rows of the updated 'data' DataFrame
print(data.head())

#DAY 3 
# So in day 3 of our project:
# We'll be creating a machine leearning model for the given data set.


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Assuming 'data' is a pandas DataFrame containing your dataset

# Extract column names from the DataFrame
temp_col = data.columns.to_list()
print("Original Columns:", temp_col, '\n')

# Create a new list of columns with a different order
new_col = temp_col[:2] + temp_col[3:]
new_col.append(temp_col[2])
print("New Columns:", new_col, '\n')

# Reorder the columns in the DataFrame
data = data.reindex(columns=new_col)
print("DataFrame with Reordered Columns:")
print(data.head())

# Standardize the data using StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)
print("Standardized Data:")
print(data[0])

# Split the data into training and testing sets
x = data[:, :-1]
y = data[:, -1]
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.1, random_state=69)

# Impute missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(x_train_imputed, y_train)
y_pred = model.predict(x_test_imputed)

# Define a function to print regression metrics
def metrics(y_true, y_pred):
    print(f'RMSE:', mean_squared_error(y_true, y_pred) ** 0.5)
    print(f'R_Squared value:', r2_score(y_true, y_pred))

# Print regression metrics for Linear Regression
print("Linear Regression Metrics:")
metrics(y_test, y_pred)

# Define a function to calculate accuracy
def accuracy(y_true, y_pred):
    errors = abs(y_true - y_pred)
    mape = 100 * np.mean(errors / y_true)
    accuracy = 100 - mape
    return accuracy

# Print accuracy for Linear Regression
print(f'Accuracy: {accuracy(y_test, y_pred)}')

# Train a Random Forest Regressor model
model_random_forest = RandomForestRegressor(n_estimators=500, min_samples_split=3)
model_random_forest.fit(x_train_imputed, y_train)
y_pred_random_forest = model_random_forest.predict(x_test_imputed)

# Print regression metrics for Random Forest Regressor
print("Random Forest Regression Metrics:")
metrics(y_test, y_pred_random_forest)

# Print accuracy for Random Forest Regressor
print(f'Accuracy: {accuracy(y_test, y_pred_random_forest)}')
