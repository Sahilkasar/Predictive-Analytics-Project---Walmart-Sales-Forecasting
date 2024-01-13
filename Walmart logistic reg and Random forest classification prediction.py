#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df1 = pd.read_csv(r"C:\Users\Sahil\Documents\Python\Walmart Sales data set\features.csv")


# In[3]:


df = df1.copy()


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


# Checking for the null values , since there are no null values ,we can move ahead with the analysis

df.isnull()


# In[8]:


df['MarkDown1'].nunique()


# In[ ]:





# In[9]:


# Since the date time column is in object form , converting it into the date time format
df['Date'] = pd.to_datetime(df['Date'] , dayfirst= True)
df['week'] = df['Date'].dt.isocalendar().week
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year


# In[10]:


df.info()


# #### Let's start the EDA 

# Sales Prediction:
# 
# If your dataset is related to retail or stores, you might want to predict sales. You could explore the relationship between sales and various factors such as temperature, fuel prices, CPI, and markdowns.
# Understanding Seasonal Trends:
# 
# Analyze the data to identify seasonal trends, especially around holidays. This could help in planning inventory, marketing strategies, and staffing.
# Impact of Markdowns on Sales:
# 
# Investigate the impact of markdowns on sales. Analyze whether markdowns lead to increased sales and if there's a correlation between the timing of markdowns and changes in sales.
# Store Performance Analysis:
# 
# Assess the performance of different stores. Identify factors that contribute to higher or lower sales, and understand the variation in key metrics across stores.
# Effect of Economic Indicators:
# 
# Explore how economic indicators such as CPI and unemployment rate relate to sales. This can help in understanding the sensitivity of sales to economic conditions.
# Optimizing Pricing Strategies:
# 
# If your dataset includes pricing information, you could explore optimal pricing strategies. Analyze the relationship between prices, sales, and other factors.
# Employee Staffing Optimization:
# 
# Understand if there's a correlation between unemployment rates and store performance. This information can be used to optimize employee staffing levels.

# In[11]:


df.info()


# In[12]:


df['Store'].unique()


# In[13]:


df['Date'].unique()


# In[14]:


# Missing data heat map
plt.figure(figsize = (10,5))
sns.heatmap(df.isnull() , cmap = 'viridis',cbar =False)
plt.title('missing data heatmap')
plt.show()


# In[15]:


# Corelation matrix
#Create indicator variables for missingness 
missing_indicators = pd.DataFrame(np.where(df.isnull(),1,0),columns = df.columns)
#concatinate missing indicators with original data set
df_with_indicators = pd.concat([df , missing_indicators.add_suffix('missing')],axis = 1)
# Calculate the correlation matrix
correlation_matrix = df_with_indicators.corr()

#visulaize the corelation matrix
plt.figure(figsize=(25,10))
sns.heatmap(correlation_matrix , cmap = 'coolwarm',annot=True)
plt.title('correlation matrix with missing indicators')
plt.show()


# In[23]:


df["CPI"].isnull().sum()


# In[36]:


df.info()


# In[24]:


# Dropping the markdown columns since they are 50% empty
# filling the null values in remaining columns with mean 
df.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],inplace=True , axis= 1)
df.fillna(df.mean() , inplace = True)


# In[25]:


df.head(2)


# In[34]:


sns.histplot(df.Store)
plt.title('Distribution of store')
plt.xlabel('Store number')
plt.ylabel('frequency')
plt.show()
plt.show()


# In[32]:


sns.histplot(df.Fuel_Price , bins = 30 , kde = True , color = 'skyblue' )
plt.title('Distribution of fule price')
plt.xlabel('fuel price')
plt.ylabel('frequency')
plt.show()


# In[39]:


sns.boxplot( x= 'IsHoliday' , y = 'CPI' , data = df)
plt.title('Box plot of CPI by Holiday')
plt.show()


# In[54]:


sns.lineplot(x = 'year',y='Unemployment',data = df)
plt.title('Unemployment over time')
plt.show()


# In[46]:


sns.violinplot(x = 'IsHoliday', y = 'Unemployment',data = df)
plt.show()


# In[55]:


sns.lineplot( x = 'year' , y = 'CPI' , data =df)
plt.title('CPI over years')
plt.show()


# In[50]:


df.head(1)


# In[57]:


from pandas.plotting import scatter_matrix
select_columns = ['Temperature','Fuel_Price','CPI','Unemployment']
scatter_matrix(df[select_columns])
plt.show()


# In[60]:


plt.figure(figsize=(16, 6))
sns.countplot(x = 'Store',data = df)
plt.title('Count of records for each store')
plt.show()


# ## Logistic regression model

# ###  IQR detection

# In[72]:


# Select numeric columns for outlier detection
numeric_columns = df.select_dtypes(include = ['float','int64']).columns

# Set up subplots
fig , axes = plt.subplots(nrows=len(numeric_columns) , ncols = 1 , figsize=(10,5*len(numeric_columns)))
fig.subplots_adjust(hspace=0.5)
                         
# Iterate through numeric columns and create boxplots
for i, column in enumerate(numeric_columns):
                         sns.boxplot(x = df[column],ax = axes[i])
                         axes[i].set_title(f'boxplot for {column}')

plt.show()


# In[74]:


# Removing outliers through z score
from scipy.stats import zscore

# select numeric columns to remove the outliers
numeric_columns = df.select_dtypes(include=['float','int64']).columns

#Calculate Z score for numeric columns
z_scores = zscore(df[numeric_columns])

#define a threshold for zscore
threshold = 3

#Create a mask to identify outliers
outlier_mask = (abs(z_scores)< threshold).all(axis = 1)

#Remove outliers
df_no_outliers = df[outlier_mask]

# Display the shape before and after removing the outliers
print(f'Shape before removing outliers:{df.shape}')
print(f'Shape after removing outliers:{df_no_outliers.shape}')
      
      


# ## Creating the Logistic regression model

# In[75]:


# Importing necessary liabraires
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix , classification_report


# In[83]:


# Selecting the target column

target = df['IsHoliday']
features = df.drop(['IsHoliday','Date'],axis = 1)

#Splitting the data into train and test split
X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state=42)

#Standardize scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#initialize the logistic regression model
logreg = LogisticRegression(random_state=42)

#Fit the model to the training data
logreg.fit(X_train_scaled , y_train)

#Make prediction on the test set
y_pred = logreg.predict(X_test_scaled)

#Evaluate the model
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nclassification Report:")
print(classification_report(y_test , y_pred))


# 
# Key observations:
# - The model has high accuracy (93%), but this is mostly due to the imbalance in the classes, with the majority being non-holidays.
# - Precision for the positive class (holiday) is 0, indicating that the model predicted no true positives among the instances it classified as holidays.
# - Recall for the positive class is also 0, indicating that the model missed all instances of actual holidays.
# - The F1-score for the positive class is 0, reflecting the poor performance in predicting holidays.
# 
# These results suggest that the model may need improvement, especially in handling the minority class (holidays). we might consider strategies like oversampling the minority class, adjusting class weights, or exploring other algorithms to improve performance on the positive class. Additionally, further feature engineering or parameter tuning could be beneficial.
# 

# In[86]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , classification_report

target = ['IsHoliday']
features = df.drop(['IsHoliday','Date'],axis =1)

print(len(target),len(features))


# In[90]:


#Need to balance the class weight 
from sklearn.utils.class_weight import compute_class_weight

#compute class weight
class_weights = compute_class_weight('balanced',classes=[False , True] , y=y_train)

#Create a dictonary to pass an argument to the logistic regression model

class_weight_dict = {False:class_weights[0],True:class_weights[1]}

#initialize the logostic regression model with class weight
logreg = LogisticRegression(random_state=42 , class_weight= class_weight_dict)

#fit the model for training data 
logreg.fit(X_train_scaled , y_train)

#Make prediction on the test set
y_pred = logreg.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test , y_pred))
print("\nClassification Report:")
print(classification_report(y_test , y_pred))


# Interpretation:
# 
# The model has higher precision for the negative class (IsHoliday=False), indicating that when it predicts an absence of holiday, it is usually correct.
# The recall for the positive class (IsHoliday=True) has increased, suggesting better performance in identifying instances of holidays.
# The overall accuracy has improved with the introduction of class weights.

# In[93]:


# Performing on another algorithm Random Forest Classifier
#Splitting into train test split
X_train , X_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state=42)

#Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#initialize the random forest
rf_classifier = RandomForestClassifier(random_state=42)

#Fit the model 
rf_classifier.fit(X_train_scaled , y_train)

#Make prediction
y_pred_rf = rf_classifier.predict(X_test_scaled)


#Evaluate the random forest

print("Confusion Matrix:")
print(confusion_matrix(y_test , y_pred_rf))
print("\nClassification_Report:")
print(classification_report(y_test , y_pred_rf))


# In summary, the model appears to be performing exceptionally well on the provided dataset, achieving perfect predictions on both classes. However, it's crucial to assess its generalization performance on unseen data and consider potential overfitting to the training set.

# In[ ]:




