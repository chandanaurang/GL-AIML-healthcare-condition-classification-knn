
# coding: utf-8

# #  Import and warehouse data:
# 

# In[146]:


#importing libraies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split # Sklearn package's randomized data splitting function


# In[98]:


columns = ['P_incidence', 'P_tilt', 'L_angle', 'S_slope', 'P_radius', 'S_Degree', 'Class']
df1=pd.read_csv('Part1 - Normal (1).csv')
df1.head()


# In[20]:


columns = ['P_incidence', 'P_tilt', 'L_angle', 'S_slope', 'P_radius', 'S_Degree', 'Class']
df2 = pd.read_csv('Part1 - Type_H (1).csv')
df2.head()


# In[18]:


columns = ['P_incidence', 'P_tilt', 'L_angle', 'S_slope', 'P_radius', 'S_Degree', 'Class']
df3 = pd.read_csv('Part1 - Type_S (1).csv')
df3.head()


# In[139]:


df=pd.concat([df1,df2,df3], ignore_index=True)


# In[39]:


df.head(10)


# In[28]:


df.describe()


# In[29]:


df.shape # Check number of columns and rows in data frame


# In[47]:


df.index


# In[32]:


df.columns


# In[48]:


df.dtypes


# In[50]:


df.count()


# In[30]:


df.info()


# #  Data cleansing:
# 

# In[52]:


df.isnull().sum()


# In[36]:


df.groupby(columns).size()


# #  Data analysis & visualisation:
# 

# In[104]:


df.describe().transpose() #descriptive statistics


# In[108]:


#univariate analysis, taking one varriable
df = df['P_incidence']


# In[109]:


len(df)


# In[111]:


df.isnull().sum() #cheching missing values


# In[112]:


plt.hist(df,bins=50)


# In[113]:


sns.distplot(df)


# In[114]:


sns.distplot(df, hist=False) # adding an argument to plot only frequency polygon


# In[116]:


sns.violinplot(df) # plots a violin plt using the seaborn package.


# In[117]:


plt.figure(figsize=(20,10)) # makes the plot wider
plt.hist(df, color='g') # plots a simple histogram
plt.axvline(df.mean(), color='m', linewidth=1)
plt.axvline(df.median(), color='b', linestyle='dashed', linewidth=1)
plt.axvline(df.mode()[0], color='w', linestyle='dashed', linewidth=1)


# In[118]:


#cumulative distribution
sns.distplot(df, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))


# # Bivariate plot

# In[140]:


df.head()


# In[149]:


sns.scatterlot(df['P_incidence'], df['P_tilt'])  # Plots the scatter plot using two variables


# # multivariate plot
# 

# In[63]:


import seaborn as sns
sns.pairplot(df)


# In[150]:


df.corr()  # displays the correlation between every possible pair of attributes as a dataframe


# In[151]:


sns.heatmap(df.corr(), annot=True)  # plot the correlation coefficients as a heatmap


# In[97]:


# Logistic Model training and Predicting

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X = df[['P_incidence', 'P_tilt', 'L_angle', 'S_slope', 'P_radius', 'S_Degree']]
y = df['class']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 100)

model = LogisticRegression()
model.fit(train_X, train_y)

prediction = model.predict(test_X)


# # Data preprocessing

# In[157]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer = imputer.fit(df.iloc[:,:-1])
imputed_df = imputer.transform(df.iloc[:,:-1].values)
df.iloc[:,:-1] = imputed_df

iris = df


# In[158]:


iris.iloc[:,6].unique()


# In[159]:


iris.head()


# In[160]:


from sklearn.preprocessing import LabelEncoder
class_label_encoder = LabelEncoder()

iris.iloc[:,-1] = class_label_encoder.fit_transform(iris.iloc[:,-1])


# In[161]:


iris.head()


# In[163]:


iris.corr()


# In[164]:


iris.var()


# In[165]:


splt = pd.scatter_matrix(iris, c=iris.iloc[:,-1], figsize=(20, 20), marker='o')


# In[167]:


#performing train-test split
import numpy as np
from sklearn.cross_validation import train_test_split

# Transform data into features and target
X = np.array(iris.ix[:, 1:6]) 
y = np.array(iris['Class'])

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# In[168]:


print(X_train.shape)
print(y_train.shape)


# In[169]:


print(X_test.shape)
print(y_test.shape)


# In[170]:


#using KNN classifier to build a model
# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors = 3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, y_pred))

# instantiate learning model (k = 5)
knn = KNeighborsClassifier(n_neighbors=5)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, y_pred))

# instantiate learning model (k = 9)
knn = KNeighborsClassifier(n_neighbors=9)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, y_pred))


# In[171]:


#cross-validation
# creating odd list of K for KNN
myList = list(range(1,20))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))


# In[172]:


# empty list that will hold accuracy scores
ac_scores = []

# perform accuracy metrics for values from 1,3,5....19
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # predict the response
    y_pred = knn.predict(X_test)
    # evaluate accuracy
    scores = accuracy_score(y_test, y_pred)
    ac_scores.append(scores)

# changing to misclassification error
MSE = [1 - x for x in ac_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)


# In[173]:


import matplotlib.pyplot as plt
# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# In[174]:


# Predicting results using Test data set
pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)


# In[ ]:


## from the above the result ,We have improved the results by fine-tuning the number of neighbors. Also, the decision boundary by KNN now is much smoother and is able to generalize well on test data.

