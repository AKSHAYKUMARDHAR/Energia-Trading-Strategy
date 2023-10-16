#!/usr/bin/env python
# coding: utf-8


# 

# In[1]:


# There was missing values in 26/03/23 wind data at 1:00 and 1:30 . It was imputed by the average wind at 00:00,00:30,2:00 ,2:30 in excel file.
import pandas as pd
GB=pd.read_excel("C:\\Users\\AKSHAY KUMAR DHAR\\Downloads\\datafiles\\GB Prices.xlsx")
DNW=pd.read_excel("C:\\Users\\AKSHAY KUMAR DHAR\\Downloads\\datafiles\\Wind and Demand.xlsx")
gas=pd.read_excel("C:\\Users\\AKSHAY KUMAR DHAR\\Downloads\\datafiles\\Historic Gas Prices.xlsx")
NIV=pd.read_excel("C:\\Users\\AKSHAY KUMAR DHAR\\Downloads\\datafiles\\Prices and NIV.xlsx")


# # EDA
# AUTHOR: DOYEL GOSH
# 

# In[3]:


GB = GB.rename(columns={'Start Time': 'Start Time 30 Minute Period'})
GB=pd.DataFrame(GB)
GB=GB.drop('Start Date Time',axis=1)
GB


# In[4]:


#AUTHOR: DOYEL Ghosh
DNW.loc[DNW['Demand (MW)']==0]


# In[5]:


#AUTHOR: DOYEL Ghosh
import numpy as np
# create a copy of the demand column
demand_imputed = DNW['Demand (MW)'].copy()

# loop over each row in the DataFrame
for i, row in DNW.iterrows():
    # check if the demand value is 0
    if row['Demand (MW)'] == 0:
        # find the last and next 5 non-zero demand values
        last_nonzero = max(i-5, 0)
        next_nonzero = min(i+5, len(DNW)-1)
        last_nonzero_demand = demand_imputed[last_nonzero:i][demand_imputed[last_nonzero:i] != 0]
        next_nonzero_demand = demand_imputed[i+1:next_nonzero+1][demand_imputed[i+1:next_nonzero+1] != 0]
        # calculate the mean of the non-zero demand values
        mean_demand = np.mean(np.concatenate((last_nonzero_demand, next_nonzero_demand)))
        # impute the demand value with the mean
        demand_imputed[i] = mean_demand

# replace the demand column in the original DataFrame with the imputed values
DNW['Demand (MW)'] = demand_imputed
DNW


# In[6]:


#AUTHOR: DOYEL Ghosh
DNW['Actual Demand(MW)'] = DNW.iloc[:,3] - DNW.iloc[:,4]
DNW


# In[7]:


#AUTHOR: DOYEL Ghosh
# merge the two datasets based on the date and time columns
merged_df = pd.merge(NIV,GB, on=['Trade Date','Start Time 30 Minute Period'], how='left')

# sort the merged dataset by date and time
#merged_df = merged_df.sort_values(['Trade Date','Start Time 30 Minute Period'])
merged_df


# In[8]:


#AUTHOR: DOYEL GOSH
merged_df1=pd.merge(merged_df,DNW, on=['Start Date','Trade Date','Start Time 30 Minute Period'], how='left')
merged_df1


# In[2]:


#AUTHOR: DOYEL Ghosh
gas = gas.rename(columns={'Date': 'Start Date'})
# Conversion rate
conv_rate = 1.14

# Convert and replace Gas Price from £/Therm to €/Therm
gas['Gas Price £/Therm'] = gas['Gas Price £/Therm'] * conv_rate

# Convert Therm to MWh
conv_rate_mwh = 0.0293
gas['Gas Price £/Therm'] =gas['Gas Price £/Therm'] / conv_rate_mwh

gas =gas.rename(columns={'Gas Price £/Therm': 'Gas Price (€/MWh)'})

# Convert the 'Date' column in df1 to datetime data type and match the format
gas['Start Date'] = pd.to_datetime(gas['Start Date'], format='%Y-%m-%d')
# Convert the 'Date' column in df2 to datetime data type
merged_df1['Start Date'] = pd.to_datetime(merged_df1['Start Date'])

# Merge the two data sets based on the 'Date' column
df=merged_df2= pd.merge(merged_df1,gas, on='Start Date',how='left')
# sort the merged dataset by date and time
#merged_df2 = merged_df2.sort_values(['Start Date'])

# Print the merged data set
merged_df2.to_csv('Energia.csv',index=False)
df=pd.DataFrame(df)
df


# # Market Price Over Time

# In[10]:


#AUTHOR: DOYEL Ghosh
import matplotlib.pyplot as plt

fig, axs = plt.subplots(5, 1, figsize=(10, 15))

axs[0].plot(df['Trade Date'], df['DAM Market Price (€/MWh)'])
axs[0].set_xlabel('Trade Date')
axs[0].set_ylabel('DAM Market Price (€/MWh)')
axs[0].set_title('DAM Market Price over Time')

axs[1].plot(df['Trade Date'], df['IDA1 Market Price (€/MWh)'])
axs[1].set_xlabel('Trade Date')
axs[1].set_ylabel('IDA1 Market Price (€/MWh)')
axs[1].set_title('IDA1 Market Price over Time')

axs[2].plot(df['Trade Date'], df['IDA2 Market Price (€/MWh)'])
axs[2].set_xlabel('Trade Date')
axs[2].set_ylabel('IDA2 Market Price (€/MWh)')
axs[2].set_title('IDA2 Market Price over Time')

axs[3].plot(df['Trade Date'], df['IDA3 Market Price (€/MWh)'])
axs[3].set_xlabel('Trade Date')
axs[3].set_ylabel('IDA3 Market Price (€/MWh)')
axs[3].set_title('IDA3 Market Price over Time')

axs[4].plot(df['Trade Date'], df['BM Market Price (€/MWh)'])
axs[4].set_xlabel('Trade Date')
axs[4].set_ylabel('BM Market Price (€/MWh)')
axs[4].set_title('BM Market Price over Time')

plt.tight_layout()
plt.show()


# # Market Price and Wind Relation

# In[11]:


#AUTHOR: DOYEL Ghosh
import matplotlib.pyplot as plt

# Create a figure and subplots
#fig, axs = plt.subplots(1, 5, figsize=(15, 3))
fig, axs = plt.subplots(5, 1, figsize=(10, 15))
# Iterate over the markets and create a scatter plot for each one
markets = ['DAM', 'IDA1', 'IDA2', 'IDA3', 'BM']
for i, market in enumerate(markets):
    axs[i].scatter(df['Actual Wind (MW)'], df[f'{market} Market Price (€/MWh)'])
    axs[i].set_xlabel('Actual Wind (MW)')
    axs[i].set_ylabel(f'{market} Market Price (€/MWh)')
    axs[i].set_title(f'{market} Market Price vs. Actual Wind')

# Adjust spacing between subplots and display the figure
fig.tight_layout()
plt.show()


# # Market Price vs. Demand

# In[12]:


#AUTHOR: DOYEL Ghosh
import matplotlib.pyplot as plt

# Create a figure and subplots
#fig, axs = plt.subplots(1, 5, figsize=(15, 3))
fig, axs = plt.subplots(5, 1, figsize=(10, 15))
# Iterate over the markets and create a scatter plot for each one
markets = ['DAM', 'IDA1', 'IDA2', 'IDA3', 'BM']
for i, market in enumerate(markets):
    axs[i].scatter(df['Demand (MW)'], df[f'{market} Market Price (€/MWh)'])
    axs[i].set_xlabel('Demand (MW)')
    axs[i].set_ylabel(f'{market} Market Price (€/MWh)')
    axs[i].set_title(f'{market} Market Price vs. Demand (MW)')

# Adjust spacing between subplots and display the figure
fig.tight_layout()
plt.show()


# # Market Prices Season wise

# In[13]:


#AUTHOR: DOYEL Ghosh
# Add a 'Season' column based on the month
df['Month'] = pd.to_datetime(df['Trade Date'], format='%d/%m/%Y').dt.month
df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [1, 2, 12] else
                                 'Spring' if x in [3, 4, 5] else
                                 'Summer' if x in [6, 7, 8] else
                                 'Autumn')

# Create a pivot table to show the mean market prices for each season and each market
mean_prices = pd.pivot_table(df, index='Season', values=['DAM Market Price (€/MWh)', 'IDA1 Market Price (€/MWh)', 
                                                         'IDA2 Market Price (€/MWh)', 'IDA3 Market Price (€/MWh)',
                                                         'BM Market Price (€/MWh)'], aggfunc='mean')

# Print the pivot table
mean_prices


# In[14]:


#AUTHOR: DOYEL Ghosh
# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.15
opacity = 0.8
index = np.arange(len(mean_prices))

for i, col in enumerate(mean_prices.columns):
    ax.bar(index + (i * bar_width), mean_prices[col], bar_width, alpha=opacity, label=col)

ax.set_xlabel('Season')
ax.set_ylabel('Mean Price (€/MWh)')
ax.set_title('Mean Market Price by Season')
ax.set_xticks(index + (bar_width * 2.5))
ax.set_xticklabels(mean_prices.index)
ax.legend()

plt.show()


# # Mean Market Prices by Half Hour Period

# In[15]:


#AUTHOR: DOYEL Ghosh
# Group the data by the half hour period and calculate the mean BM Market Price for each group
grouped_df = df.groupby('Start Time 30 Minute Period')['BM Market Price (€/MWh)'].median()

# Create a bar chart of the mean BM Market Price for each half hour period
grouped_df.plot(kind='bar')
plt.xlabel('Start Time 30 Minute Period')
plt.ylabel('Mean BM Market Price (€/MWh)')
plt.title('Mean BM Market Price by Half Hour Period')
plt.show()


# In[16]:


#AUTHOR: DOYEL Ghosh
# Group the data by the half hour period and calculate the mean BM Market Price for each group
grouped_df = df.groupby('Start Time 30 Minute Period')['DAM Market Price (€/MWh)'].median()

# Create a bar chart of the mean BM Market Price for each half hour period
grouped_df.plot(kind='bar')
plt.xlabel('Start Time 30 Minute Period')
plt.ylabel('Median DAM Market Price (€/MWh)')
plt.title('DAM Market Price by Half Hour Period')
plt.show()


# In[17]:


#AUTHOR: DOYEL Ghosh
# Group the data by the half hour period and calculate the mean BM Market Price for each group
grouped_df = df.groupby('Start Time 30 Minute Period')['IDA1 Market Price (€/MWh)'].mean()

# Create a bar chart of the mean BM Market Price for each half hour period
grouped_df.plot(kind='bar')
plt.xlabel('Start Time 30 Minute Period')
plt.ylabel('Median IDA1 Market Price (€/MWh)')
plt.title('IDA1 Market Price (€/MWh)')
plt.show()


# In[18]:


#AUTHOR: DOYEL Ghosh
# Group the data by the half hour period and calculate the mean BM Market Price for each group
grouped_df = df.groupby('Start Time 30 Minute Period')['IDA2 Market Price (€/MWh)'].mean()

# Create a bar chart of the mean BM Market Price for each half hour period
grouped_df.plot(kind='bar')
plt.xlabel('Start Time 30 Minute Period')
plt.ylabel('Median IDA2 Market Price (€/MWh)')
plt.title('Mean BM Market Price by Half Hour Period')
plt.show()


# In[19]:


#AUTHOR: DOYEL Ghosh
# Group the data by the half hour period and calculate the mean BM Market Price for each group
grouped_df = df.groupby('Start Time 30 Minute Period')['IDA3 Market Price (€/MWh)'].mean()

# Create a bar chart of the mean BM Market Price for each half hour period
grouped_df.plot(kind='bar')
plt.xlabel('Start Time 30 Minute Period')
plt.ylabel('Median IDA3 Market Price (€/MWh)')
plt.title('IDA3 Market Price by Half Hour Period')
plt.show()


# # Market wise correlation heatmap

# In[20]:


#AUTHOR: DOYEL Ghosh
# Calculate the correlation matrix
import seaborn as sns

corr_matrix = df[['IDA1 Market Price (€/MWh)', 'GB Price (€/MWh)', 'Actual Demand(MW)','Gas Price (€/MWh)']].corr()
# Visualize the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
corr_matrix = df[['IDA2 Market Price (€/MWh)', 'GB Price (€/MWh)', 'Actual Demand(MW)', 'Gas Price (€/MWh)']].corr()
# Visualize the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

corr_matrix = df[['IDA3 Market Price (€/MWh)', 'GB Price (€/MWh)', 'Actual Demand(MW)', 'Gas Price (€/MWh)']].corr()
# Visualize the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

corr_matrix = df[['BM Market Price (€/MWh)', 'GB Price (€/MWh)', 'Actual Demand(MW)', 'Gas Price (€/MWh)']].corr()
# Visualize the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()



# In[21]:


#AUTHOR: DOYEL Ghosh
import matplotlib.pyplot as plt

plt.plot(df['Trade Date'], df['Gas Price (€/MWh)'])
plt.xlabel('Trade Date')
plt.ylabel('Gas Price (€/MWh)')
plt.title('Gas Price over Trade Date')
plt.show()
# Impute the gas price column NaN values with average of next 10 days
df['Gas Price (€/MWh)'].fillna(57.95, inplace=True)


# In[22]:


#AUTHOR: DOYEL Ghosh
df['Trade Time'] = pd.to_datetime(df['Trade Date'] + ' ' + df['Start Time 30 Minute Period'])
df.set_index('Trade Time', inplace=True)
df


# # Multiple Linear Regression Model

# In[23]:


#AUTHOR: DOYEL Ghosh
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Extract the input features and target variable
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)', 'GB Price (€/MWh)']]
y = df[['DAM Market Price (€/MWh)','IDA1 Market Price (€/MWh)']]

# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit the linear regression model to the training data
model = LinearRegression().fit(X_train, y_train)

# Make predictions on the testing data using the trained model
y_pred = model.predict(X_test)

# Calculate the mean squared error, R-squared score, and adjusted R-squared score of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
n = len(X_test)
k = len(X.columns)
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Print the results
print(f"Mean Squared Error: {mse}")
print(f"R-squared score: {r2}")
print(f"Adjusted R-squared score: {adj_r2}")

# Calculate the residuals (difference between predicted and actual values)
residuals = y_test - y_pred

# Create a residual plot
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[24]:


#AUTHOR: DOYEL Ghosh
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)', 'GB Price (€/MWh)']]
y_pred = model.predict(X)
predictions_df = pd.DataFrame(y_pred, index=df.index, columns=['DAM Predicted','IDA1 Predicted'])
pd.DataFrame(predictions_df)


# In[25]:


#AUTHOR: DOYEL Ghosh
# drop the observations where 'IDA2 Market Price (€/MWh)' column has NaN values
df1=df.dropna(subset=['IDA2 Market Price (€/MWh)'])
df2=df.dropna(subset=['IDA3 Market Price (€/MWh)'])


# In[26]:


#AUTHOR: DOYEL Ghosh
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Extract the input features and target variable
X = df1[['Gas Price (€/MWh)', 'Actual Demand(MW)', 'GB Price (€/MWh)']]
y = df1[['IDA2 Market Price (€/MWh)']]

# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit the linear regression model to the training data
model = LinearRegression().fit(X_train, y_train)

# Make predictions on the testing data using the trained model
y_pred = model.predict(X_test)

# Calculate the mean squared error, R-squared score, and adjusted R-squared score of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
n = len(X_test)
k = len(X.columns)
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Print the results
print(f"Mean Squared Error: {mse}")
print(f"R-squared score: {r2}")
print(f"Adjusted R-squared score: {adj_r2}")

# Calculate the residuals (difference between predicted and actual values)
residuals = y_test - y_pred

# Create a residual plot
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[27]:


#AUTHOR: DOYEL Ghosh
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)', 'GB Price (€/MWh)']]
y_pred = model.predict(X)
predictions_df = pd.DataFrame(y_pred, index=df.index, columns=['IDA2 Predicted'])
pd.DataFrame(predictions_df)


# In[28]:


#AUTHOR: DOYEL Ghosh
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Extract the input features and target variable
X = df2[['Gas Price (€/MWh)', 'Actual Demand(MW)']]
y = df2[['IDA3 Market Price (€/MWh)']]

# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit the linear regression model to the training data
model = LinearRegression().fit(X_train, y_train)

# Make predictions on the testing data using the trained model
y_pred = model.predict(X_test)

# Calculate the mean squared error, R-squared score, and adjusted R-squared score of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
n = len(X_test)
k = len(X.columns)
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Print the results
print(f"Mean Squared Error: {mse}")
print(f"R-squared score: {r2}")
print(f"Adjusted R-squared score: {adj_r2}")

# Calculate the residuals (difference between predicted and actual values)
residuals = y_test - y_pred

# Create a residual plot
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[29]:


#AUTHOR: DOYEL Ghosh
X = df2[['Gas Price (€/MWh)', 'Actual Demand(MW)']]
y_pred = model.predict(X)
predictions_df = pd.DataFrame(y_pred, index=df2.index, columns=['IDA3 Predicted'])
pd.DataFrame(predictions_df)


# In[30]:


#AUTHOR: DOYEL Ghosh
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Extract the input features and target variable
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)', 'GB Price (€/MWh)']]
y = df[['BM Market Price (€/MWh)']]

# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit the linear regression model to the training data
model = LinearRegression().fit(X_train, y_train)

# Make predictions on the testing data using the trained model
y_pred = model.predict(X_test)

# Calculate the mean squared error, R-squared score, and adjusted R-squared score of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
n = len(X_test)
k = len(X.columns)
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Print the results
print(f"Mean Squared Error: {mse}")
print(f"R-squared score: {r2}")
print(f"Adjusted R-squared score: {adj_r2}")

# Calculate the residuals (difference between predicted and actual values)
residuals = y_test - y_pred

# Create a residual plot
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# In[31]:


#AUTHOR: DOYEL Ghosh
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)','GB Price (€/MWh)']]
y_pred = model.predict(X)
predictions_df = pd.DataFrame(y_pred, index=df.index, columns=['BM Predicted'])
pd.DataFrame(predictions_df)


# # Random Forest Regressor

# In[32]:


#AUTHOR: DOYEL Ghosh
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# Split data into features (X) and target variable (y)
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)','GB Price (€/MWh)']]
y = df['BM Market Price (€/MWh)']

# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model performance metrics
print("Mean Squared Error (MSE): ", mse)
print("Coefficient of determination (R^2): ", r2)

import matplotlib.pyplot as plt

# Plot the actual and predicted market prices
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Observation')
plt.ylabel('BM Market Price (€/MWh)')
plt.title('Actual vs Predicted BM Market Price')
plt.legend()
plt.show()


# In[33]:


#AUTHOR: DOYEL Ghosh
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)', 'GB Price (€/MWh)']]
y_pred =rf_model.predict(X)
predictions_BM = pd.DataFrame(y_pred, index=df.index, columns=['BM Predicted'])
pd.DataFrame(predictions_BM)


# In[34]:


#AUTHOR: DOYEL Ghosh
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# Split data into features (X) and target variable (y)
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)','GB Price (€/MWh)']]
y = df['DAM Market Price (€/MWh)']
# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model performance metrics
print("Mean Squared Error (MSE): ", mse)
print("Coefficient of determination (R^2): ", r2)
# Plot the actual and predicted market prices
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Observation')
plt.ylabel('DAM Market Price (€/MWh)')
plt.title('Actual vs Predicted DAM Market Price')
plt.legend()
plt.show()


# In[35]:


#AUTHOR: DOYEL Ghosh
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)', 'GB Price (€/MWh)']]
y_pred =rf_model.predict(X)
predictions_DAM = pd.DataFrame(y_pred, index=df.index, columns=['DAM Predicted'])
pd.DataFrame(predictions_DAM )


# In[36]:


#AUTHOR: DOYEL Ghosh
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# Split data into features (X) and target variable (y)
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)','GB Price (€/MWh)']]
y = df['IDA1 Market Price (€/MWh)']
# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model performance metrics
print("Mean Squared Error (MSE): ", mse)
print("Coefficient of determination (R^2): ", r2)
# Plot the actual and predicted market prices
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Observation')
plt.ylabel('IDA1 Market Price (€/MWh)')
plt.title('Actual vs Predicted IDA1 Market Price')
plt.legend()
plt.show()


# In[37]:


#AUTHOR: DOYEL  Ghosh
X = df[['Gas Price (€/MWh)', 'Actual Demand(MW)', 'GB Price (€/MWh)']]
y_pred =rf_model.predict(X)
predictions_IDA1= pd.DataFrame(y_pred, index=df.index, columns=['IDA1 Predicted'])
pd.DataFrame(predictions_IDA1)


# In[38]:


#AUTHOR DOYEL Ghosh
# Split data into features (X) and target variable (y)
X = df1[['Gas Price (€/MWh)', 'Actual Demand(MW)','GB Price (€/MWh)']]
y = df1['IDA2 Market Price (€/MWh)']
# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model performance metrics
print("Mean Squared Error (MSE): ", mse)
print("Coefficient of determination (R^2): ", r2)


# In[39]:


#AUTHOR DOYEL Ghosh
X = df1[['Gas Price (€/MWh)', 'Actual Demand(MW)', 'GB Price (€/MWh)']]
y_pred =rf_model.predict(X)
predictions_IDA2 = pd.DataFrame(y_pred, index=df1.index, columns=['IDA2 Predicted'])
pd.DataFrame(predictions_IDA2)


# In[40]:


#AUTHOR: DOYEL Ghosh
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Split data into features (X) and target variable (y)
X = df2[['Gas Price (€/MWh)', 'Actual Demand(MW)']]
y = df2['IDA3 Market Price (€/MWh)']
# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model performance metrics
print("Mean Squared Error (MSE): ", mse)
print("Coefficient of determination (R^2): ", r2)



import matplotlib.pyplot as plt

# Plot the actual and predicted market prices
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Observation')
plt.ylabel('IDA3 Market Price (€/MWh)')
plt.title('Actual vs Predicted IDA3 Market Price')
plt.legend()
plt.show()


# In[41]:


#AUTHOR: DOYEL Ghosh
X = df2[['Gas Price (€/MWh)', 'Actual Demand(MW)']]
y_pred = rf_model .predict(X)
predictions_IDA3 = pd.DataFrame(y_pred, index=df2.index, columns=['IDA3 Predicted'])
pd.DataFrame(predictions_IDA3)


# In[51]:


#AUTHOR: DOYEL Ghosh
merged_table = pd.merge(predictions_DAM,predictions_IDA1,how='outer', on='Trade Time')
merged_table =pd.merge(merged_table,predictions_IDA2,how='outer', on='Trade Time')
merged_table =pd.merge(merged_table,predictions_IDA3,how='outer', on='Trade Time')
data=pd.merge(merged_table,predictions_BM,how='outer', on='Trade Time')
data['Start Time 30 Minute Period']=df['Start Time 30 Minute Period']
data['Trade Date']=df['Trade Date']
data_copy=data

data


# # Minimum price of every half an hour and market offering minimum price 

# In[43]:


#AUTHOR: DOYEL Ghosh
data_copy=data
data_copy=data_copy.reset_index()


final_price, market = [],[]
for index, row in data_copy.iterrows():
    final_price.append(row[['DAM Predicted', 'IDA1 Predicted','IDA2 Predicted','IDA3 Predicted','BM Predicted']].min())
    if final_price[index]==row['BM Predicted']:
        market.append('BM')
    elif final_price[index]==row['DAM Predicted']:
        market.append('DAM')
    elif final_price[index]==row['IDA1 Predicted']:
        market.append('IDA1')
    elif final_price[index]==row['IDA2 Predicted']: 
        market.append('IDA2')
    else:
        market.append('IDA3')

            
data_copy['min_price']=final_price
data_copy['best_market']=market
data_copy


# # Set a threshold to 100 €/MWh 

# In[44]:


#AUTHOR: DOYEL Ghosh
#Set a threshold to 100 €/MWh to DM market and if the price in dm market crosses 100 €/MWh then for that half BM market will not be considered
data=data.reset_index()

final_price, market = [],[]
for index, row in data.iterrows():
    if row['DAM Predicted']>=100:
        final_price.append(row[['DAM Predicted', 'IDA1 Predicted','IDA2 Predicted','IDA3 Predicted']].min())
    else:
        final_price.append(row[['DAM Predicted', 'IDA1 Predicted','IDA2 Predicted','IDA3 Predicted','BM Predicted']].min())
    if final_price[index]==row['BM Predicted']:
        market.append('BM')
    elif final_price[index]==row['DAM Predicted']:
        market.append('DAM')
    elif final_price[index]==row['IDA1 Predicted']:
        market.append('IDA1')
    elif final_price[index]==row['IDA2 Predicted']: 
        market.append('IDA2')
    else:
        market.append('IDA3')

            
data['min_price']=final_price
data['best_market']=market
data


# In[45]:


#AUTHOR: DOYEL Ghosh
import matplotlib.pyplot as plt

# Group by target class and count the number of occurrences
class_counts = data_copy.groupby('best_market')['best_market'].count()

# Create a bar plot
fig, ax = plt.subplots()
class_counts.plot(kind='bar', ax=ax)
ax.set_xlabel('best_market')
ax.set_ylabel('Count')
plt.show()


# In[46]:


#AUTHOR: DOYEL Ghosh
import matplotlib.pyplot as plt

# Group by target class and count the number of occurrences
class_counts = data.groupby('best_market')['best_market'].count()

# Create a bar plot
fig, ax = plt.subplots()
class_counts.plot(kind='bar', ax=ax)
ax.set_xlabel('best_market')
ax.set_ylabel('Count')
plt.show()


# In[50]:


#AUTHOR: DOYEL Ghosh
each_day=data.groupby('Trade Date')
profit_df=pd.DataFrame()
profit_df['Trade Date']=data['Trade Date'].unique()
best_price, DAM_price,profit_precent=[],[],[]
for i in data['Trade Date'].unique():
        best_price.append((each_day.get_group(i)['min_price'].sum())*100/2)
        DAM_price.append((each_day.get_group(i)['DAM Predicted'].sum())*100/2)
profit_df['Best Price']=best_price
profit_df['DAM Price']=DAM_price
profit_df['Profit']=profit_df['DAM Price']-profit_df['Best Price']
profit_df

