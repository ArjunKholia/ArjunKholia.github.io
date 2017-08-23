---
title: "Service Disruption Prediction"
layout: post
date: 2017-08-22 22:10
tag: jekyll
image: https://koppl.in/indigo/assets/images/jekyll-logo-light-solid.png
headerImage: true
projects: true
hidden: true # don't count this post in blog pagination
description: "This is a simple and minimalist template for Jekyll for those who likes to eat noodles."
category: project
author: Arjun Kholia
externalLink: false
---

# Service Disruption Prediction project

## Libraries


```python
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Import the Data


```python
train = pd.read_csv("train.csv")

event = pd.read_csv("event_type.csv")

severity = pd.read_csv("severity_type.csv")

log_feature = pd.read_csv("log_feature.csv")

resource = pd.read_csv("resource_type.csv")

```


```python
train.shape,event.shape,severity.shape,log_feature.shape,resource.shape
```




    ((7008, 3), (31170, 2), (18552, 2), (58671, 3), (21076, 2))



## Import and Read Test data


```python
test = pd.read_excel("test_pbl.csv",header = None)
test.columns = ['id','location']
test.shape
```




    (373, 2)




```python
print(train.head())
print(event.head())
print(severity.head())
print(log_feature.head())
print(resource.head())
print(test.head())
```

          id      location  fault_severity
    0  14121  location 118               1
    1   9320   location 91               0
    2  14394  location 152               1
    3   8218  location 931               1
    4  14804  location 120               0
         id     event_type
    0  6597  event_type 11
    1  8011  event_type 15
    2  2597  event_type 15
    3  5022  event_type 15
    4  5022  event_type 11
         id    severity_type
    0  6597  severity_type 2
    1  8011  severity_type 2
    2  2597  severity_type 2
    3  5022  severity_type 1
    4  6852  severity_type 1
         id  log_feature  volume
    0  6597   feature 68       6
    1  8011   feature 68       7
    2  2597   feature 68       1
    3  5022  feature 172       2
    4  5022   feature 56       1
         id    resource_type
    0  6597  resource_type 8
    1  8011  resource_type 8
    2  2597  resource_type 8
    3  5022  resource_type 8
    4  6852  resource_type 8
          id      location
    0  11695  location 244
    1  15412  location 845
    2   1972  location 922
    3     64  location 921
    4   2897  location 878


## Extract the integers from columns and remove strings


```python
train['location'] = train['location'].str.extract('(\d+)', expand=True)
event['event_type'] = event['event_type'].str.extract('(\d+)', expand=True)
severity['severity_type'] = severity['severity_type'].str.extract('(\d+)', expand=True)
log_feature['log_feature'] = log_feature['log_feature'].str.extract('(\d+)', expand=True)
resource['resource_type'] = resource['resource_type'].str.extract('(\d+)', expand=True)
test['location'] = test['location'].str.extract('(\d+)', expand=True)
```

## Data Preprocessing and Merging to create a single record (CAR)


```python
tt = train.append(test)
tt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7381 entries, 0 to 372
    Data columns (total 3 columns):
    fault_severity    7008 non-null float64
    id                7381 non-null int64
    location          7381 non-null object
    dtypes: float64(1), int64(1), object(1)
    memory usage: 230.7+ KB



```python
l = [event, severity, log_feature, resource]
df = reduce(lambda left,right: pd.merge(left,right,on='id'), l)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 146423 entries, 0 to 146422
    Data columns (total 6 columns):
    id               146423 non-null int64
    event_type       146423 non-null object
    severity_type    146423 non-null object
    log_feature      146423 non-null object
    volume           146423 non-null int64
    resource_type    146423 non-null object
    dtypes: int64(2), object(4)
    memory usage: 7.8+ MB



```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>event_type</th>
      <th>severity_type</th>
      <th>log_feature</th>
      <th>volume</th>
      <th>resource_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6597</td>
      <td>11</td>
      <td>2</td>
      <td>68</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8011</td>
      <td>15</td>
      <td>2</td>
      <td>68</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2597</td>
      <td>15</td>
      <td>2</td>
      <td>68</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5022</td>
      <td>15</td>
      <td>1</td>
      <td>172</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5022</td>
      <td>15</td>
      <td>1</td>
      <td>56</td>
      <td>1</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
x = tt.drop('fault_severity',1)
df = pd.merge(df,x,on = 'id')
df = df[['id','location','event_type','resource_type','severity_type','log_feature','volume']]
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>location</th>
      <th>event_type</th>
      <th>resource_type</th>
      <th>severity_type</th>
      <th>log_feature</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8011</td>
      <td>1</td>
      <td>15</td>
      <td>8</td>
      <td>2</td>
      <td>68</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2588</td>
      <td>1</td>
      <td>15</td>
      <td>8</td>
      <td>1</td>
      <td>82</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2588</td>
      <td>1</td>
      <td>15</td>
      <td>8</td>
      <td>1</td>
      <td>201</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2588</td>
      <td>1</td>
      <td>15</td>
      <td>8</td>
      <td>1</td>
      <td>80</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2588</td>
      <td>1</td>
      <td>15</td>
      <td>8</td>
      <td>1</td>
      <td>203</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 61839 entries, 0 to 61838
    Data columns (total 7 columns):
    id               61839 non-null int64
    location         61839 non-null object
    event_type       61839 non-null object
    resource_type    61839 non-null object
    severity_type    61839 non-null object
    log_feature      61839 non-null object
    volume           61839 non-null int64
    dtypes: int64(2), object(5)
    memory usage: 3.8+ MB


## Check for duplicate data


```python
df[df.duplicated()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>location</th>
      <th>event_type</th>
      <th>resource_type</th>
      <th>severity_type</th>
      <th>log_feature</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
l = df.columns
for a in l[:len(l)-1]:
    print('number of unique ' + a + ' : ' + str(df[a].nunique()))   
```

    number of unique id : 7381
    number of unique location : 929
    number of unique event_type : 49
    number of unique resource_type : 10
    number of unique severity_type : 5
    number of unique log_feature : 331


## Create new feature columns


```python
#total volume per id
total_volume = log_feature.groupby(['id'],as_index = False)[['volume']].sum()
total_volume.columns = ['id','total_volume']
```


```python
df.severity_type.value_counts()
```




    1    36571
    2    24260
    4      920
    5       55
    3       33
    Name: severity_type, dtype: int64




```python
total_volume['total_volume'].describe()
```




    count    18552.000000
    mean        30.629905
    std         77.755460
    min          1.000000
    25%          3.000000
    50%          8.000000
    75%         25.000000
    max       1649.000000
    Name: total_volume, dtype: float64



### Normalise the Volume Column


```python
from sklearn import preprocessing

# Create x, where x the 'volume' column's values as floats
x = total_volume['total_volume'].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
volume_normal = pd.DataFrame(x_scaled)
volume_normal.columns = ['volume_normal']

#concat the normalizsed volume column to main dataframe
total_volume = pd.concat([total_volume,volume_normal],axis = 1)
```

    C:\Users\Arjun\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:321: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
    C:\Users\Arjun\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py:356: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)



```python
loc_vol_features =pd.merge(total_volume[['id','volume_normal']],tt,on='id',how = 'inner')

loc_vol_features.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>volume_normal</th>
      <th>fault_severity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7381.000000</td>
      <td>7381.000000</td>
      <td>7008.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9264.649099</td>
      <td>0.018702</td>
      <td>0.450342</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5349.290176</td>
      <td>0.047893</td>
      <td>0.665220</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4647.000000</td>
      <td>0.001214</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9222.000000</td>
      <td>0.004248</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13885.000000</td>
      <td>0.014563</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>18550.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
features = pd.merge(severity,loc_vol_features,on='id',how='inner')

features['location'] = features['location'].astype(int)
features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7381 entries, 0 to 7380
    Data columns (total 5 columns):
    id                7381 non-null int64
    severity_type     7381 non-null object
    volume_normal     7381 non-null float64
    fault_severity    7008 non-null float64
    location          7381 non-null int32
    dtypes: float64(2), int32(1), int64(1), object(1)
    memory usage: 317.2+ KB



```python
features.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>severity_type</th>
      <th>volume_normal</th>
      <th>fault_severity</th>
      <th>location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8011</td>
      <td>2</td>
      <td>0.003641</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2588</td>
      <td>1</td>
      <td>0.020024</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4848</td>
      <td>1</td>
      <td>0.020631</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6914</td>
      <td>1</td>
      <td>0.006068</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5337</td>
      <td>1</td>
      <td>0.014563</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Visualization


```python
#Total Volume distribution in features
facet = sns.FacetGrid( loc_vol_features ,  aspect=4)
facet.map( sns.kdeplot , 'volume_normal' , shade= True )
facet.set( xlim=( 0 , loc_vol_features['volume_normal'].max()) )
facet.add_legend()
```

    C:\Users\Arjun\Anaconda3\lib\site-packages\statsmodels\nonparametric\kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j





    <seaborn.axisgrid.FacetGrid at 0x2363834b7f0>




![png](output_31_2.png)



```python
# Plot of fault types across different locations and ids
fig, ax1 = plt.subplots(figsize=(7,7))
ax1.scatter(features.loc[features.fault_severity.isnull(),'location'],features.loc[features.fault_severity.isnull(),'id'],alpha=0.8,marker = 's',color='k',s=12,label = 'test')
ax1.scatter(features.loc[features.fault_severity==0,'location'],features.loc[features.fault_severity==0,'id'],alpha=0.8,color='g',label = 'fault_severity 0',s= 16)
ax1.scatter(features.loc[features.fault_severity==1,'location'],features.loc[features.fault_severity==1,'id'],alpha=0.8,color='r',label = 'fault_severity 1',s= 16)
ax1.scatter(features.loc[features.fault_severity==2,'location'],features.loc[features.fault_severity==2,'id'],alpha=0.8,color='b',label = 'fault_severity 2',s= 16)
ax1.set_xlim((-100,1200))
ax1.set_ylim((0,19000))
ax1.set_xlabel('location')
ax1.set_ylabel('id')
ax1.legend(bbox_to_anchor=(1.1, 0.5), loc='upper left')
```




    <matplotlib.legend.Legend at 0x236387b26a0>




![png](output_32_1.png)



```python
#Plot of Severity types across different id and locations for each fault types
#Fault severity 2 occurs largely for location greater than 600
#Fault severity 1 has a gap between location 450-550
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
g = sns.FacetGrid(features, col="fault_severity",size = 5,hue = 'severity_type',palette =flatui )
g = (g.map(plt.scatter, "location", "id", edgecolor="w").add_legend())
```


![png](output_33_0.png)



```python
#Heatmap and Stacked Barplot for Severity type and Fault type

fault_type_total = pd.crosstab(features['severity_type'],features['fault_severity'])
# Normalize the cross tab to sum to 1 & Plot:
fault_type_ntotal = fault_type_total.div(fault_type_total.sum(1).astype(float), axis=0)


fig = plt.figure(figsize = (8,4))
ax1= fig.add_subplot(121)
ax2= fig.add_subplot(122)

sns.heatmap(fault_type_total,cmap='RdYlGn_r', linewidths=0.5, annot=True,ax=ax1)

plot1 = fault_type_ntotal.plot(kind='barh',stacked=True,title='Fault type by severity types',ax=ax2,colormap = 'viridis')
plt.xlabel('Fault type')
plt.ylabel('severity types')
plt.legend(bbox_to_anchor=(1.1, 0.5), loc='upper left', ncol=1)

plt.show()
```


![png](output_34_0.png)



```python
# Plot of Volumne vs location
fig, ax1 = plt.subplots(figsize=(6,6))
ax1.scatter(features.loc[features.fault_severity.isnull(),'location'],features.loc[features.fault_severity.isnull(),'volume_normal'],alpha=0.6,marker = 's',color='k',s=8,label = 'test')
ax1.scatter(features.loc[features.fault_severity==0,'location'],features.loc[features.fault_severity==0,'volume_normal'],alpha=0.6,color='g',label = 'fault_severity 0',s= 16)
ax1.scatter(features.loc[features.fault_severity==1,'location'],features.loc[features.fault_severity==1,'volume_normal'],alpha=0.6,color='r',label = 'fault_severity 1',s= 16)
ax1.scatter(features.loc[features.fault_severity==2,'location'],features.loc[features.fault_severity==2,'volume_normal'],alpha=0.6,color='b',label = 'fault_severity 2',s= 16)
ax1.set_xlim((-100,1200))
ax1.set_ylim((0,0.5))
ax1.set_xlabel('location')
ax1.set_ylabel('volume_normal')
ax1.legend(bbox_to_anchor=(1.1, 0.5), loc='upper left')
```




    <matplotlib.legend.Legend at 0x2363973ff60>




![png](output_35_1.png)



```python
feature_cor = features.corr()
sns.heatmap(feature_cor,cmap='RdYlGn_r', linewidths=0.5, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x236382e7a20>




![png](output_36_1.png)



```python
#Adding a number feature for location
features['num'] = features.groupby(['location']).cumcount()+1
features.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>severity_type</th>
      <th>volume_normal</th>
      <th>fault_severity</th>
      <th>location</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8011</td>
      <td>2</td>
      <td>0.003641</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2588</td>
      <td>1</td>
      <td>0.020024</td>
      <td>0.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4848</td>
      <td>1</td>
      <td>0.020631</td>
      <td>0.0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6914</td>
      <td>1</td>
      <td>0.006068</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5337</td>
      <td>1</td>
      <td>0.014563</td>
      <td>0.0</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
features.tail(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>severity_type</th>
      <th>volume_normal</th>
      <th>fault_severity</th>
      <th>location</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7371</th>
      <td>8390</td>
      <td>1</td>
      <td>0.001820</td>
      <td>1.0</td>
      <td>998</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7372</th>
      <td>8972</td>
      <td>1</td>
      <td>0.006675</td>
      <td>0.0</td>
      <td>998</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7373</th>
      <td>14916</td>
      <td>1</td>
      <td>0.001820</td>
      <td>1.0</td>
      <td>998</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7374</th>
      <td>13670</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>998</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7375</th>
      <td>4196</td>
      <td>1</td>
      <td>0.000607</td>
      <td>1.0</td>
      <td>999</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7376</th>
      <td>6288</td>
      <td>1</td>
      <td>0.001214</td>
      <td>1.0</td>
      <td>999</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7377</th>
      <td>13296</td>
      <td>1</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>999</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7378</th>
      <td>8114</td>
      <td>2</td>
      <td>0.001820</td>
      <td>0.0</td>
      <td>999</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7379</th>
      <td>878</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>999</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7380</th>
      <td>4464</td>
      <td>1</td>
      <td>0.001214</td>
      <td>0.0</td>
      <td>999</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Plotting the new number feature vs location
#We see higher average number of different ids for same location for locations greater than 600
#Location 400 to 600 contains most of the faults with type zero
fig, ax1 = plt.subplots(figsize=(7,7))
ax1.scatter(features.loc[features.fault_severity.isnull(),'location'],features.loc[features.fault_severity.isnull(),'num'],alpha=0.6,marker = 's',color='k',s=8,label = 'test')
ax1.scatter(features.loc[features.fault_severity==0,'location'],features.loc[features.fault_severity==0,'num'],alpha=0.6,color='g',label = 'fault_severity 0',s= 16)
ax1.scatter(features.loc[features.fault_severity==1,'location'],features.loc[features.fault_severity==1,'num'],alpha=0.6,color='r',label = 'fault_severity 1',s= 16)
ax1.scatter(features.loc[features.fault_severity==2,'location'],features.loc[features.fault_severity==2,'num'],alpha=0.6,color='b',label = 'fault_severity 2',s= 16)
ax1.set_xlim((-100,1200))
ax1.set_ylim((0,90))
ax1.set_xlabel('location')
ax1.set_ylabel('num')
ax1.legend(bbox_to_anchor=(1.1, 0.5), loc='upper left')
```




    <matplotlib.legend.Legend at 0x23639cfc898>




![png](output_39_1.png)



```python
g = sns.FacetGrid(features, col="fault_severity",size = 5)
g = g.map(plt.scatter, "location", "num", edgecolor="w",alpha = 0.7,color = 'g')
```


![png](output_40_0.png)



```python
#Create a location_grp feature
def location_grp(data):
    if data < 450:
        return 'lower_location'
    elif data <550:
        return 'middle_location'
    else:
        return 'higher_location'
features['location_grp'] = features['location'].apply(location_grp)
```


```python
g = sns.FacetGrid(features, col="fault_severity",size = 5)
g = g.map(plt.scatter, "num", "volume_normal", edgecolor="w",alpha= 0.6,color = 'r')
```


![png](output_42_0.png)



```python
resource_feature = pd.merge(tt,resource,on='id',how ='inner')
resource_feature['resource_type'] = resource_feature['resource_type'].astype(int)
resource_feature['location'] = resource_feature['location'].astype(int)
resource_feature.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fault_severity</th>
      <th>id</th>
      <th>location</th>
      <th>resource_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>14121</td>
      <td>118</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>9320</td>
      <td>91</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>14394</td>
      <td>152</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>8218</td>
      <td>931</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>14804</td>
      <td>120</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>14804</td>
      <td>120</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1080</td>
      <td>664</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>9731</td>
      <td>640</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>15505</td>
      <td>122</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>3443</td>
      <td>263</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Resource type 8,2 make 91% of data
x = resource['resource_type'].value_counts()
y = x/x.sum()
y
```




    8     0.487189
    2     0.423135
    6     0.027614
    7     0.023629
    4     0.015658
    9     0.009015
    3     0.006880
    10    0.003464
    1     0.002752
    5     0.000664
    Name: resource_type, dtype: float64




```python
#Resource 2 covers region from 100-500
#Resource 8 covers from 0-100 and 600-1100
g = sns.FacetGrid(resource_feature, col="fault_severity",size = 5,hue = 'resource_type',palette={'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'})
g = g.map(plt.scatter, "location", "id",edgecolor="w").add_legend()
```


![png](output_45_0.png)



```python
#Create a resource_grp feature
def resource_grp(data):
    if data == '8':
        return 'resource_8'
    elif data == '2':
        return 'resource_2'
    else:
        return 'other_resource'
df['resource_grp'] = df['resource_type'].apply(resource_grp)
features.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>severity_type</th>
      <th>volume_normal</th>
      <th>fault_severity</th>
      <th>location</th>
      <th>num</th>
      <th>location_grp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8011</td>
      <td>2</td>
      <td>0.003641</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>lower_location</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2588</td>
      <td>1</td>
      <td>0.020024</td>
      <td>0.0</td>
      <td>1</td>
      <td>2</td>
      <td>lower_location</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4848</td>
      <td>1</td>
      <td>0.020631</td>
      <td>0.0</td>
      <td>1</td>
      <td>3</td>
      <td>lower_location</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6914</td>
      <td>1</td>
      <td>0.006068</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
      <td>lower_location</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5337</td>
      <td>1</td>
      <td>0.014563</td>
      <td>0.0</td>
      <td>1</td>
      <td>5</td>
      <td>lower_location</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
l = df.columns
for a in l[:len(l)]:
    print('number of unique ' + a + ' : ' + str(df[a].nunique()))   
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 61839 entries, 0 to 61838
    Data columns (total 8 columns):
    id               61839 non-null int64
    location         61839 non-null object
    event_type       61839 non-null object
    resource_type    61839 non-null object
    severity_type    61839 non-null object
    log_feature      61839 non-null object
    volume           61839 non-null int64
    resource_grp     61839 non-null object
    dtypes: int64(2), object(6)
    memory usage: 4.2+ MB
    number of unique id : 7381
    number of unique location : 929
    number of unique event_type : 49
    number of unique resource_type : 10
    number of unique severity_type : 5
    number of unique log_feature : 331
    number of unique volume : 254
    number of unique resource_grp : 3



```python
df_dummy = pd.get_dummies(df,drop_first=True)

df_dummy.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>volume</th>
      <th>location_10</th>
      <th>location_100</th>
      <th>location_1000</th>
      <th>location_1002</th>
      <th>location_1005</th>
      <th>location_1006</th>
      <th>location_1007</th>
      <th>location_1008</th>
      <th>...</th>
      <th>log_feature_91</th>
      <th>log_feature_92</th>
      <th>log_feature_94</th>
      <th>log_feature_95</th>
      <th>log_feature_96</th>
      <th>log_feature_97</th>
      <th>log_feature_98</th>
      <th>log_feature_99</th>
      <th>resource_grp_resource_2</th>
      <th>resource_grp_resource_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8011</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2588</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2588</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2588</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2588</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1323 columns</p>
</div>




```python
final_df = pd.DataFrame(df_dummy.groupby('id',as_index=False).sum())
final_df.info()
final_df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7381 entries, 0 to 7380
    Columns: 1323 entries, id to resource_grp_resource_8
    dtypes: float64(1321), int64(2)
    memory usage: 74.6 MB





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>volume</th>
      <th>location_10</th>
      <th>location_100</th>
      <th>location_1000</th>
      <th>location_1002</th>
      <th>location_1005</th>
      <th>location_1006</th>
      <th>location_1007</th>
      <th>location_1008</th>
      <th>...</th>
      <th>log_feature_91</th>
      <th>log_feature_92</th>
      <th>log_feature_94</th>
      <th>log_feature_95</th>
      <th>log_feature_96</th>
      <th>log_feature_97</th>
      <th>log_feature_98</th>
      <th>log_feature_99</th>
      <th>resource_grp_resource_2</th>
      <th>resource_grp_resource_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>34</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>32</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1323 columns</p>
</div>




```python
final_df = pd.merge(features[['id','location_grp','num','volume_normal']],final_df,on='id')
```


```python
final_df['num'] = final_df['num'].astype(str)
final_df.info()
final_df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7381 entries, 0 to 7380
    Columns: 1326 entries, id to resource_grp_resource_8
    dtypes: float64(1322), int64(2), object(2)
    memory usage: 74.7+ MB





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>location_grp</th>
      <th>num</th>
      <th>volume_normal</th>
      <th>volume</th>
      <th>location_10</th>
      <th>location_100</th>
      <th>location_1000</th>
      <th>location_1002</th>
      <th>location_1005</th>
      <th>...</th>
      <th>log_feature_91</th>
      <th>log_feature_92</th>
      <th>log_feature_94</th>
      <th>log_feature_95</th>
      <th>log_feature_96</th>
      <th>log_feature_97</th>
      <th>log_feature_98</th>
      <th>log_feature_99</th>
      <th>resource_grp_resource_2</th>
      <th>resource_grp_resource_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8011</td>
      <td>lower_location</td>
      <td>1</td>
      <td>0.003641</td>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2588</td>
      <td>lower_location</td>
      <td>2</td>
      <td>0.020024</td>
      <td>68</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4848</td>
      <td>lower_location</td>
      <td>3</td>
      <td>0.020631</td>
      <td>70</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6914</td>
      <td>lower_location</td>
      <td>4</td>
      <td>0.006068</td>
      <td>22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5337</td>
      <td>lower_location</td>
      <td>5</td>
      <td>0.014563</td>
      <td>50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1326 columns</p>
</div>




```python
final_df_dummy = pd.get_dummies(final_df[['id','location_grp','num']],drop_first=True)
final_df = final_df.drop(['location_grp','num','volume'],axis = 1)
final_df = pd.merge(final_df,final_df_dummy,on='id')
final_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>volume_normal</th>
      <th>location_10</th>
      <th>location_100</th>
      <th>location_1000</th>
      <th>location_1002</th>
      <th>location_1005</th>
      <th>location_1006</th>
      <th>location_1007</th>
      <th>location_1008</th>
      <th>...</th>
      <th>num_78</th>
      <th>num_79</th>
      <th>num_8</th>
      <th>num_80</th>
      <th>num_81</th>
      <th>num_82</th>
      <th>num_83</th>
      <th>num_84</th>
      <th>num_85</th>
      <th>num_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8011</td>
      <td>0.003641</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2588</td>
      <td>0.020024</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4848</td>
      <td>0.020631</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6914</td>
      <td>0.006068</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5337</td>
      <td>0.014563</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1409 columns</p>
</div>




```python
final_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7381 entries, 0 to 7380
    Columns: 1409 entries, id to num_9
    dtypes: float64(1322), int64(1), uint8(86)
    memory usage: 75.2 MB


## Split the Original Train and Test Dataframe


```python
final_df_train = pd.merge(train[['id','fault_severity']],final_df,on='id',how = 'inner')
final_df_test = pd.merge(test[['id']],final_df,on='id',how = 'inner')
final_df_train.info()
final_df_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7008 entries, 0 to 7007
    Columns: 1410 entries, id to num_9
    dtypes: float64(1322), int64(2), uint8(86)
    memory usage: 71.4 MB
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 373 entries, 0 to 372
    Columns: 1409 entries, id to num_9
    dtypes: float64(1322), int64(1), uint8(86)
    memory usage: 3.8 MB



```python
y = final_df_train['fault_severity']
X = final_df_train.drop('fault_severity',1)
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((5606, 1409), (1402, 1409), (5606,), (1402,))




```python
"""a = y_train['fault_severity'].value_counts()
a1 = a/a.sum()
b = y_test['fault_severity'].value_counts()
b1 = b/b.sum()
c = y['fault_severity'].value_counts()
c1 = c/c.sum()
print(a1)
print(b1)
print(c1)"""
```




    "a = y_train['fault_severity'].value_counts()\na1 = a/a.sum()\nb = y_test['fault_severity'].value_counts()\nb1 = b/b.sum()\nc = y['fault_severity'].value_counts()\nc1 = c/c.sum()\nprint(a1)\nprint(b1) \nprint(c1)"




```python
# Outlier detection
from collections import Counter
def detect_outliers(data,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(data[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(data[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step )].index
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers   
```


```python
outliers=detect_outliers(final_df,2,['volume_normal'])
final_df.loc[outliers]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>volume_normal</th>
      <th>location_10</th>
      <th>location_100</th>
      <th>location_1000</th>
      <th>location_1002</th>
      <th>location_1005</th>
      <th>location_1006</th>
      <th>location_1007</th>
      <th>location_1008</th>
      <th>...</th>
      <th>num_78</th>
      <th>num_79</th>
      <th>num_8</th>
      <th>num_80</th>
      <th>num_81</th>
      <th>num_82</th>
      <th>num_83</th>
      <th>num_84</th>
      <th>num_85</th>
      <th>num_9</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 1409 columns</p>
</div>



## Dimension Reduction - Feature Selection


```python
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
```


```python
# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=1000, n_jobs=4)

# Train the classifier
clf.fit(X_train, y_train)

```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=1000, n_jobs=4, oob_score=False,
                random_state=None, verbose=0, warm_start=False)




```python
# Print number of columns before initially
a = 1
for feature in zip(X_train.columns, clf.feature_importances_):
    a+=1
print(a)
```

    1410



```python
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.05
sfm = SelectFromModel(clf, threshold=0.0001)

# Train the selector
sfm.fit(X_train, y_train)

# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

```


```python
# Print number of columns before after feature selection
a=1
new_features = []
for feature_list_index in sfm.get_support(indices=True):
    new_features.append(X_train.columns[feature_list_index])
    a = a+1
print(a)
```

    807



```python
X_train2 = X_train.loc[:,new_features]
X_test2 = X_test.loc[:,new_features]
X_train2 = pd.DataFrame(X_train2)
```


```python
X_train2.info()
X_train2.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 5606 entries, 1246 to 2732
    Columns: 806 entries, id to num_9
    dtypes: float64(727), int64(1), uint8(78)
    memory usage: 31.6 MB





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>volume_normal</th>
      <th>location_100</th>
      <th>location_1007</th>
      <th>location_1008</th>
      <th>location_1010</th>
      <th>location_1011</th>
      <th>location_1014</th>
      <th>location_1015</th>
      <th>location_1016</th>
      <th>...</th>
      <th>num_70</th>
      <th>num_71</th>
      <th>num_72</th>
      <th>num_73</th>
      <th>num_74</th>
      <th>num_78</th>
      <th>num_8</th>
      <th>num_82</th>
      <th>num_84</th>
      <th>num_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1246</th>
      <td>10398</td>
      <td>0.003641</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>675</th>
      <td>13296</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>381</th>
      <td>8423</td>
      <td>0.001820</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4957</th>
      <td>4419</td>
      <td>0.000607</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2337</th>
      <td>6020</td>
      <td>0.004854</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 806 columns</p>
</div>



# Modelling

## Algorithms


```python
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import cross_validation, metrics   #Additional scklearn functions

```


```python
def modelfit(alg, dtrain, dtest, performCV=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain, dtest)

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain, dtest, cv=cv_folds, scoring='accuracy')

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtest.values, dtrain_predictions))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
```


```python
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, X_train2, y_train)
```


    Model Report
    Accuracy : 0.7797
    CV Score : Mean - 0.7358185 | Std - 0.007307129 | Min - 0.7216771 | Max - 0.7421945


## Parameter Tuning in Gradient Boosting


```python
#Tuning the parameters
param_test = {
               'n_estimators':range(500,601,50),
               'min_samples_leaf':range(20,71,10),
               'max_depth': [8,9,10]
              }
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate = 0.1,min_samples_split = 500,max_features='sqrt',subsample=0.8,random_state=10),
param_grid = param_test, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train2,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```

    C:\Users\Arjun\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:667: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)





    ([mean: 0.74527, std: 0.00705, params: {'max_depth': 8, 'min_samples_leaf': 20, 'n_estimators': 500},
      mean: 0.74349, std: 0.00942, params: {'max_depth': 8, 'min_samples_leaf': 20, 'n_estimators': 550},
      mean: 0.74367, std: 0.00898, params: {'max_depth': 8, 'min_samples_leaf': 20, 'n_estimators': 600},
      mean: 0.74402, std: 0.00700, params: {'max_depth': 8, 'min_samples_leaf': 30, 'n_estimators': 500},
      mean: 0.74081, std: 0.00644, params: {'max_depth': 8, 'min_samples_leaf': 30, 'n_estimators': 550},
      mean: 0.73956, std: 0.00724, params: {'max_depth': 8, 'min_samples_leaf': 30, 'n_estimators': 600},
      mean: 0.74420, std: 0.00812, params: {'max_depth': 8, 'min_samples_leaf': 40, 'n_estimators': 500},
      mean: 0.74509, std: 0.00807, params: {'max_depth': 8, 'min_samples_leaf': 40, 'n_estimators': 550},
      mean: 0.74509, std: 0.00794, params: {'max_depth': 8, 'min_samples_leaf': 40, 'n_estimators': 600},
      mean: 0.74616, std: 0.00950, params: {'max_depth': 8, 'min_samples_leaf': 50, 'n_estimators': 500},
      mean: 0.74527, std: 0.00899, params: {'max_depth': 8, 'min_samples_leaf': 50, 'n_estimators': 550},
      mean: 0.74563, std: 0.00784, params: {'max_depth': 8, 'min_samples_leaf': 50, 'n_estimators': 600},
      mean: 0.74170, std: 0.00702, params: {'max_depth': 8, 'min_samples_leaf': 60, 'n_estimators': 500},
      mean: 0.74117, std: 0.00759, params: {'max_depth': 8, 'min_samples_leaf': 60, 'n_estimators': 550},
      mean: 0.74313, std: 0.00701, params: {'max_depth': 8, 'min_samples_leaf': 60, 'n_estimators': 600},
      mean: 0.73992, std: 0.00626, params: {'max_depth': 8, 'min_samples_leaf': 70, 'n_estimators': 500},
      mean: 0.73956, std: 0.00780, params: {'max_depth': 8, 'min_samples_leaf': 70, 'n_estimators': 550},
      mean: 0.73867, std: 0.00899, params: {'max_depth': 8, 'min_samples_leaf': 70, 'n_estimators': 600},
      mean: 0.74206, std: 0.01223, params: {'max_depth': 9, 'min_samples_leaf': 20, 'n_estimators': 500},
      mean: 0.74474, std: 0.01369, params: {'max_depth': 9, 'min_samples_leaf': 20, 'n_estimators': 550},
      mean: 0.74384, std: 0.01305, params: {'max_depth': 9, 'min_samples_leaf': 20, 'n_estimators': 600},
      mean: 0.74135, std: 0.00723, params: {'max_depth': 9, 'min_samples_leaf': 30, 'n_estimators': 500},
      mean: 0.74242, std: 0.00875, params: {'max_depth': 9, 'min_samples_leaf': 30, 'n_estimators': 550},
      mean: 0.74295, std: 0.00731, params: {'max_depth': 9, 'min_samples_leaf': 30, 'n_estimators': 600},
      mean: 0.74616, std: 0.00779, params: {'max_depth': 9, 'min_samples_leaf': 40, 'n_estimators': 500},
      mean: 0.74509, std: 0.00757, params: {'max_depth': 9, 'min_samples_leaf': 40, 'n_estimators': 550},
      mean: 0.74652, std: 0.00830, params: {'max_depth': 9, 'min_samples_leaf': 40, 'n_estimators': 600},
      mean: 0.74599, std: 0.00800, params: {'max_depth': 9, 'min_samples_leaf': 50, 'n_estimators': 500},
      mean: 0.74723, std: 0.00686, params: {'max_depth': 9, 'min_samples_leaf': 50, 'n_estimators': 550},
      mean: 0.74741, std: 0.00720, params: {'max_depth': 9, 'min_samples_leaf': 50, 'n_estimators': 600},
      mean: 0.74313, std: 0.00937, params: {'max_depth': 9, 'min_samples_leaf': 60, 'n_estimators': 500},
      mean: 0.74242, std: 0.00789, params: {'max_depth': 9, 'min_samples_leaf': 60, 'n_estimators': 550},
      mean: 0.74188, std: 0.00881, params: {'max_depth': 9, 'min_samples_leaf': 60, 'n_estimators': 600},
      mean: 0.74099, std: 0.00974, params: {'max_depth': 9, 'min_samples_leaf': 70, 'n_estimators': 500},
      mean: 0.74117, std: 0.00938, params: {'max_depth': 9, 'min_samples_leaf': 70, 'n_estimators': 550},
      mean: 0.74081, std: 0.00822, params: {'max_depth': 9, 'min_samples_leaf': 70, 'n_estimators': 600},
      mean: 0.74402, std: 0.01376, params: {'max_depth': 10, 'min_samples_leaf': 20, 'n_estimators': 500},
      mean: 0.74367, std: 0.01330, params: {'max_depth': 10, 'min_samples_leaf': 20, 'n_estimators': 550},
      mean: 0.74527, std: 0.01237, params: {'max_depth': 10, 'min_samples_leaf': 20, 'n_estimators': 600},
      mean: 0.74367, std: 0.00548, params: {'max_depth': 10, 'min_samples_leaf': 30, 'n_estimators': 500},
      mean: 0.74367, std: 0.00622, params: {'max_depth': 10, 'min_samples_leaf': 30, 'n_estimators': 550},
      mean: 0.74420, std: 0.00748, params: {'max_depth': 10, 'min_samples_leaf': 30, 'n_estimators': 600},
      mean: 0.74741, std: 0.00889, params: {'max_depth': 10, 'min_samples_leaf': 40, 'n_estimators': 500},
      mean: 0.74474, std: 0.00883, params: {'max_depth': 10, 'min_samples_leaf': 40, 'n_estimators': 550},
      mean: 0.74723, std: 0.00848, params: {'max_depth': 10, 'min_samples_leaf': 40, 'n_estimators': 600},
      mean: 0.74795, std: 0.00938, params: {'max_depth': 10, 'min_samples_leaf': 50, 'n_estimators': 500},
      mean: 0.74830, std: 0.00919, params: {'max_depth': 10, 'min_samples_leaf': 50, 'n_estimators': 550},
      mean: 0.74456, std: 0.00903, params: {'max_depth': 10, 'min_samples_leaf': 50, 'n_estimators': 600},
      mean: 0.74491, std: 0.00812, params: {'max_depth': 10, 'min_samples_leaf': 60, 'n_estimators': 500},
      mean: 0.74456, std: 0.00706, params: {'max_depth': 10, 'min_samples_leaf': 60, 'n_estimators': 550},
      mean: 0.74420, std: 0.00791, params: {'max_depth': 10, 'min_samples_leaf': 60, 'n_estimators': 600},
      mean: 0.73885, std: 0.01092, params: {'max_depth': 10, 'min_samples_leaf': 70, 'n_estimators': 500},
      mean: 0.73974, std: 0.01050, params: {'max_depth': 10, 'min_samples_leaf': 70, 'n_estimators': 550},
      mean: 0.73921, std: 0.00933, params: {'max_depth': 10, 'min_samples_leaf': 70, 'n_estimators': 600}],
     {'max_depth': 10, 'min_samples_leaf': 50, 'n_estimators': 550},
     0.74830293807572501)




```python
model = GradientBoostingClassifier(max_depth = 10, min_samples_leaf = 50, n_estimators= 550,learning_rate = 0.1,min_samples_split = 500,max_features='sqrt',subsample=0.8,random_state=10)
model = model.fit(X,y)
predicted = model.predict(final_df_test)
predicted_probability = model.predict_proba(final_df_test)
```


```python
predicted = pd.DataFrame(predicted)
predicted.columns = ['Predicted']
predicted.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_result = final_df_test[['id']]
final_result = pd.merge(final_result,test[['id','location']],on='id')
final_result['predicted'] = predicted['Predicted']
final_result['probability 0'] = predicted_probability[:,0:1]
final_result['probability 1'] = predicted_probability[:,1:2]
final_result['probability 2'] = predicted_probability[:,2:3]

final_result.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>location</th>
      <th>predicted</th>
      <th>probability 0</th>
      <th>probability 1</th>
      <th>probability 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11695</td>
      <td>244</td>
      <td>0</td>
      <td>0.938485</td>
      <td>0.061494</td>
      <td>0.000022</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15412</td>
      <td>845</td>
      <td>2</td>
      <td>0.080899</td>
      <td>0.138348</td>
      <td>0.780753</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1972</td>
      <td>922</td>
      <td>0</td>
      <td>0.513650</td>
      <td>0.480723</td>
      <td>0.005627</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64</td>
      <td>921</td>
      <td>0</td>
      <td>0.953237</td>
      <td>0.033334</td>
      <td>0.013428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2897</td>
      <td>878</td>
      <td>0</td>
      <td>0.927408</td>
      <td>0.067372</td>
      <td>0.005220</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_result = final_result.sort_values('probability 2',ascending=False)
final_result['location'] = final_result['location'].astype(int)
final_result.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 373 entries, 321 to 118
    Data columns (total 6 columns):
    id               373 non-null int64
    location         373 non-null int32
    predicted        373 non-null int64
    probability 0    373 non-null float64
    probability 1    373 non-null float64
    probability 2    373 non-null float64
    dtypes: float64(3), int32(1), int64(2)
    memory usage: 18.9 KB



```python
test
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11695</td>
      <td>244</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15412</td>
      <td>845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1972</td>
      <td>922</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64</td>
      <td>921</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2897</td>
      <td>878</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2708</td>
      <td>311</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12865</td>
      <td>478</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14741</td>
      <td>649</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15514</td>
      <td>430</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9361</td>
      <td>1066</td>
    </tr>
    <tr>
      <th>10</th>
      <td>17629</td>
      <td>155</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15650</td>
      <td>133</td>
    </tr>
    <tr>
      <th>12</th>
      <td>18543</td>
      <td>998</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4605</td>
      <td>7</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15922</td>
      <td>831</td>
    </tr>
    <tr>
      <th>15</th>
      <td>968</td>
      <td>255</td>
    </tr>
    <tr>
      <th>16</th>
      <td>12815</td>
      <td>921</td>
    </tr>
    <tr>
      <th>17</th>
      <td>12879</td>
      <td>1100</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8265</td>
      <td>319</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6388</td>
      <td>1052</td>
    </tr>
    <tr>
      <th>20</th>
      <td>15753</td>
      <td>921</td>
    </tr>
    <tr>
      <th>21</th>
      <td>8616</td>
      <td>1026</td>
    </tr>
    <tr>
      <th>22</th>
      <td>16349</td>
      <td>1107</td>
    </tr>
    <tr>
      <th>23</th>
      <td>4596</td>
      <td>700</td>
    </tr>
    <tr>
      <th>24</th>
      <td>17616</td>
      <td>793</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2675</td>
      <td>249</td>
    </tr>
    <tr>
      <th>26</th>
      <td>16526</td>
      <td>644</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5285</td>
      <td>1086</td>
    </tr>
    <tr>
      <th>28</th>
      <td>15968</td>
      <td>760</td>
    </tr>
    <tr>
      <th>29</th>
      <td>12645</td>
      <td>734</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>343</th>
      <td>12449</td>
      <td>946</td>
    </tr>
    <tr>
      <th>344</th>
      <td>2840</td>
      <td>193</td>
    </tr>
    <tr>
      <th>345</th>
      <td>6391</td>
      <td>128</td>
    </tr>
    <tr>
      <th>346</th>
      <td>9093</td>
      <td>766</td>
    </tr>
    <tr>
      <th>347</th>
      <td>7032</td>
      <td>300</td>
    </tr>
    <tr>
      <th>348</th>
      <td>1108</td>
      <td>345</td>
    </tr>
    <tr>
      <th>349</th>
      <td>2864</td>
      <td>755</td>
    </tr>
    <tr>
      <th>350</th>
      <td>5346</td>
      <td>229</td>
    </tr>
    <tr>
      <th>351</th>
      <td>13814</td>
      <td>810</td>
    </tr>
    <tr>
      <th>352</th>
      <td>9551</td>
      <td>825</td>
    </tr>
    <tr>
      <th>353</th>
      <td>5946</td>
      <td>561</td>
    </tr>
    <tr>
      <th>354</th>
      <td>11952</td>
      <td>495</td>
    </tr>
    <tr>
      <th>355</th>
      <td>4256</td>
      <td>504</td>
    </tr>
    <tr>
      <th>356</th>
      <td>6410</td>
      <td>919</td>
    </tr>
    <tr>
      <th>357</th>
      <td>2030</td>
      <td>925</td>
    </tr>
    <tr>
      <th>358</th>
      <td>8981</td>
      <td>643</td>
    </tr>
    <tr>
      <th>359</th>
      <td>8336</td>
      <td>902</td>
    </tr>
    <tr>
      <th>360</th>
      <td>16262</td>
      <td>504</td>
    </tr>
    <tr>
      <th>361</th>
      <td>11613</td>
      <td>478</td>
    </tr>
    <tr>
      <th>362</th>
      <td>3450</td>
      <td>444</td>
    </tr>
    <tr>
      <th>363</th>
      <td>4065</td>
      <td>238</td>
    </tr>
    <tr>
      <th>364</th>
      <td>1628</td>
      <td>224</td>
    </tr>
    <tr>
      <th>365</th>
      <td>16687</td>
      <td>1090</td>
    </tr>
    <tr>
      <th>366</th>
      <td>6813</td>
      <td>1115</td>
    </tr>
    <tr>
      <th>367</th>
      <td>10455</td>
      <td>1075</td>
    </tr>
    <tr>
      <th>368</th>
      <td>870</td>
      <td>167</td>
    </tr>
    <tr>
      <th>369</th>
      <td>18068</td>
      <td>106</td>
    </tr>
    <tr>
      <th>370</th>
      <td>14111</td>
      <td>1086</td>
    </tr>
    <tr>
      <th>371</th>
      <td>15189</td>
      <td>7</td>
    </tr>
    <tr>
      <th>372</th>
      <td>17067</td>
      <td>885</td>
    </tr>
  </tbody>
</table>
<p>373 rows × 2 columns</p>
</div>




```python
final_result
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>location</th>
      <th>predicted</th>
      <th>probability 0</th>
      <th>probability 1</th>
      <th>probability 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>321</th>
      <td>13754</td>
      <td>613</td>
      <td>2</td>
      <td>0.060348</td>
      <td>0.110666</td>
      <td>8.289860e-01</td>
    </tr>
    <tr>
      <th>17</th>
      <td>12879</td>
      <td>1100</td>
      <td>2</td>
      <td>0.092510</td>
      <td>0.091573</td>
      <td>8.159172e-01</td>
    </tr>
    <tr>
      <th>217</th>
      <td>8087</td>
      <td>684</td>
      <td>2</td>
      <td>0.141741</td>
      <td>0.068079</td>
      <td>7.901800e-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15412</td>
      <td>845</td>
      <td>2</td>
      <td>0.080899</td>
      <td>0.138348</td>
      <td>7.807527e-01</td>
    </tr>
    <tr>
      <th>358</th>
      <td>8981</td>
      <td>643</td>
      <td>2</td>
      <td>0.154993</td>
      <td>0.075405</td>
      <td>7.696017e-01</td>
    </tr>
    <tr>
      <th>187</th>
      <td>14998</td>
      <td>995</td>
      <td>2</td>
      <td>0.163348</td>
      <td>0.082871</td>
      <td>7.537807e-01</td>
    </tr>
    <tr>
      <th>230</th>
      <td>2770</td>
      <td>1100</td>
      <td>2</td>
      <td>0.162407</td>
      <td>0.091328</td>
      <td>7.462655e-01</td>
    </tr>
    <tr>
      <th>134</th>
      <td>10591</td>
      <td>834</td>
      <td>2</td>
      <td>0.161447</td>
      <td>0.128459</td>
      <td>7.100940e-01</td>
    </tr>
    <tr>
      <th>63</th>
      <td>11650</td>
      <td>763</td>
      <td>2</td>
      <td>0.166033</td>
      <td>0.131233</td>
      <td>7.027338e-01</td>
    </tr>
    <tr>
      <th>91</th>
      <td>14198</td>
      <td>962</td>
      <td>2</td>
      <td>0.175050</td>
      <td>0.133242</td>
      <td>6.917079e-01</td>
    </tr>
    <tr>
      <th>111</th>
      <td>17578</td>
      <td>704</td>
      <td>2</td>
      <td>0.145832</td>
      <td>0.166998</td>
      <td>6.871701e-01</td>
    </tr>
    <tr>
      <th>175</th>
      <td>9421</td>
      <td>1021</td>
      <td>2</td>
      <td>0.140709</td>
      <td>0.192830</td>
      <td>6.664604e-01</td>
    </tr>
    <tr>
      <th>291</th>
      <td>3626</td>
      <td>798</td>
      <td>2</td>
      <td>0.170848</td>
      <td>0.166154</td>
      <td>6.629980e-01</td>
    </tr>
    <tr>
      <th>222</th>
      <td>12126</td>
      <td>1100</td>
      <td>2</td>
      <td>0.179846</td>
      <td>0.167903</td>
      <td>6.522505e-01</td>
    </tr>
    <tr>
      <th>206</th>
      <td>16429</td>
      <td>1107</td>
      <td>2</td>
      <td>0.169739</td>
      <td>0.178452</td>
      <td>6.518094e-01</td>
    </tr>
    <tr>
      <th>263</th>
      <td>3061</td>
      <td>846</td>
      <td>2</td>
      <td>0.212980</td>
      <td>0.145279</td>
      <td>6.417411e-01</td>
    </tr>
    <tr>
      <th>342</th>
      <td>10511</td>
      <td>1008</td>
      <td>2</td>
      <td>0.254902</td>
      <td>0.104822</td>
      <td>6.402758e-01</td>
    </tr>
    <tr>
      <th>69</th>
      <td>18258</td>
      <td>810</td>
      <td>2</td>
      <td>0.271970</td>
      <td>0.094885</td>
      <td>6.331452e-01</td>
    </tr>
    <tr>
      <th>269</th>
      <td>4232</td>
      <td>1107</td>
      <td>2</td>
      <td>0.257913</td>
      <td>0.133075</td>
      <td>6.090120e-01</td>
    </tr>
    <tr>
      <th>324</th>
      <td>1787</td>
      <td>599</td>
      <td>2</td>
      <td>0.107567</td>
      <td>0.292186</td>
      <td>6.002469e-01</td>
    </tr>
    <tr>
      <th>198</th>
      <td>1065</td>
      <td>600</td>
      <td>2</td>
      <td>0.350905</td>
      <td>0.060588</td>
      <td>5.885065e-01</td>
    </tr>
    <tr>
      <th>168</th>
      <td>15339</td>
      <td>1107</td>
      <td>2</td>
      <td>0.139838</td>
      <td>0.293060</td>
      <td>5.671016e-01</td>
    </tr>
    <tr>
      <th>97</th>
      <td>15622</td>
      <td>1075</td>
      <td>2</td>
      <td>0.183046</td>
      <td>0.277036</td>
      <td>5.399180e-01</td>
    </tr>
    <tr>
      <th>243</th>
      <td>16222</td>
      <td>600</td>
      <td>2</td>
      <td>0.347750</td>
      <td>0.113556</td>
      <td>5.386944e-01</td>
    </tr>
    <tr>
      <th>29</th>
      <td>12645</td>
      <td>734</td>
      <td>2</td>
      <td>0.249508</td>
      <td>0.229630</td>
      <td>5.208621e-01</td>
    </tr>
    <tr>
      <th>351</th>
      <td>13814</td>
      <td>810</td>
      <td>2</td>
      <td>0.027153</td>
      <td>0.463968</td>
      <td>5.088783e-01</td>
    </tr>
    <tr>
      <th>114</th>
      <td>3330</td>
      <td>1042</td>
      <td>2</td>
      <td>0.222144</td>
      <td>0.271120</td>
      <td>5.067353e-01</td>
    </tr>
    <tr>
      <th>16</th>
      <td>12815</td>
      <td>921</td>
      <td>2</td>
      <td>0.162927</td>
      <td>0.333832</td>
      <td>5.032418e-01</td>
    </tr>
    <tr>
      <th>288</th>
      <td>1102</td>
      <td>600</td>
      <td>2</td>
      <td>0.323116</td>
      <td>0.183926</td>
      <td>4.929583e-01</td>
    </tr>
    <tr>
      <th>192</th>
      <td>10226</td>
      <td>798</td>
      <td>2</td>
      <td>0.232454</td>
      <td>0.276584</td>
      <td>4.909620e-01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>212</th>
      <td>2275</td>
      <td>144</td>
      <td>1</td>
      <td>0.327740</td>
      <td>0.672259</td>
      <td>1.128303e-06</td>
    </tr>
    <tr>
      <th>219</th>
      <td>1867</td>
      <td>242</td>
      <td>0</td>
      <td>0.975797</td>
      <td>0.024202</td>
      <td>1.035001e-06</td>
    </tr>
    <tr>
      <th>180</th>
      <td>4818</td>
      <td>348</td>
      <td>0</td>
      <td>0.616485</td>
      <td>0.383514</td>
      <td>9.860500e-07</td>
    </tr>
    <tr>
      <th>226</th>
      <td>17199</td>
      <td>475</td>
      <td>0</td>
      <td>0.943833</td>
      <td>0.056166</td>
      <td>9.011698e-07</td>
    </tr>
    <tr>
      <th>295</th>
      <td>13197</td>
      <td>124</td>
      <td>0</td>
      <td>0.936028</td>
      <td>0.063971</td>
      <td>8.382601e-07</td>
    </tr>
    <tr>
      <th>294</th>
      <td>14531</td>
      <td>161</td>
      <td>0</td>
      <td>0.818536</td>
      <td>0.181463</td>
      <td>7.996703e-07</td>
    </tr>
    <tr>
      <th>83</th>
      <td>13834</td>
      <td>149</td>
      <td>0</td>
      <td>0.630240</td>
      <td>0.369760</td>
      <td>7.916186e-07</td>
    </tr>
    <tr>
      <th>258</th>
      <td>11884</td>
      <td>97</td>
      <td>1</td>
      <td>0.469722</td>
      <td>0.530277</td>
      <td>7.690266e-07</td>
    </tr>
    <tr>
      <th>210</th>
      <td>18219</td>
      <td>484</td>
      <td>0</td>
      <td>0.990782</td>
      <td>0.009217</td>
      <td>7.559375e-07</td>
    </tr>
    <tr>
      <th>165</th>
      <td>7880</td>
      <td>122</td>
      <td>0</td>
      <td>0.968527</td>
      <td>0.031472</td>
      <td>7.246467e-07</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2416</td>
      <td>358</td>
      <td>0</td>
      <td>0.786333</td>
      <td>0.213666</td>
      <td>6.854999e-07</td>
    </tr>
    <tr>
      <th>81</th>
      <td>16241</td>
      <td>460</td>
      <td>0</td>
      <td>0.953871</td>
      <td>0.046129</td>
      <td>6.270073e-07</td>
    </tr>
    <tr>
      <th>37</th>
      <td>7246</td>
      <td>497</td>
      <td>0</td>
      <td>0.998231</td>
      <td>0.001768</td>
      <td>5.343139e-07</td>
    </tr>
    <tr>
      <th>143</th>
      <td>5550</td>
      <td>313</td>
      <td>0</td>
      <td>0.862620</td>
      <td>0.137379</td>
      <td>5.006856e-07</td>
    </tr>
    <tr>
      <th>31</th>
      <td>13288</td>
      <td>193</td>
      <td>0</td>
      <td>0.589177</td>
      <td>0.410822</td>
      <td>4.826170e-07</td>
    </tr>
    <tr>
      <th>156</th>
      <td>3582</td>
      <td>471</td>
      <td>0</td>
      <td>0.996831</td>
      <td>0.003168</td>
      <td>4.766915e-07</td>
    </tr>
    <tr>
      <th>40</th>
      <td>6925</td>
      <td>497</td>
      <td>0</td>
      <td>0.995884</td>
      <td>0.004115</td>
      <td>4.515839e-07</td>
    </tr>
    <tr>
      <th>101</th>
      <td>6785</td>
      <td>126</td>
      <td>0</td>
      <td>0.940878</td>
      <td>0.059121</td>
      <td>4.468179e-07</td>
    </tr>
    <tr>
      <th>51</th>
      <td>7560</td>
      <td>495</td>
      <td>0</td>
      <td>0.998080</td>
      <td>0.001920</td>
      <td>3.903395e-07</td>
    </tr>
    <tr>
      <th>148</th>
      <td>16695</td>
      <td>478</td>
      <td>0</td>
      <td>0.989939</td>
      <td>0.010061</td>
      <td>3.668348e-07</td>
    </tr>
    <tr>
      <th>267</th>
      <td>8062</td>
      <td>102</td>
      <td>0</td>
      <td>0.861038</td>
      <td>0.138962</td>
      <td>3.614613e-07</td>
    </tr>
    <tr>
      <th>211</th>
      <td>8951</td>
      <td>505</td>
      <td>0</td>
      <td>0.998066</td>
      <td>0.001934</td>
      <td>3.600606e-07</td>
    </tr>
    <tr>
      <th>238</th>
      <td>7188</td>
      <td>124</td>
      <td>0</td>
      <td>0.950656</td>
      <td>0.049344</td>
      <td>3.187127e-07</td>
    </tr>
    <tr>
      <th>95</th>
      <td>17418</td>
      <td>471</td>
      <td>0</td>
      <td>0.997432</td>
      <td>0.002568</td>
      <td>3.092161e-07</td>
    </tr>
    <tr>
      <th>170</th>
      <td>2539</td>
      <td>127</td>
      <td>0</td>
      <td>0.903944</td>
      <td>0.096056</td>
      <td>2.615446e-07</td>
    </tr>
    <tr>
      <th>68</th>
      <td>4289</td>
      <td>325</td>
      <td>0</td>
      <td>0.541246</td>
      <td>0.458754</td>
      <td>2.564295e-07</td>
    </tr>
    <tr>
      <th>155</th>
      <td>16555</td>
      <td>126</td>
      <td>0</td>
      <td>0.981138</td>
      <td>0.018862</td>
      <td>2.413048e-07</td>
    </tr>
    <tr>
      <th>345</th>
      <td>6391</td>
      <td>128</td>
      <td>0</td>
      <td>0.889596</td>
      <td>0.110404</td>
      <td>2.258314e-07</td>
    </tr>
    <tr>
      <th>354</th>
      <td>11952</td>
      <td>495</td>
      <td>0</td>
      <td>0.997629</td>
      <td>0.002371</td>
      <td>2.028331e-07</td>
    </tr>
    <tr>
      <th>118</th>
      <td>13905</td>
      <td>126</td>
      <td>0</td>
      <td>0.976771</td>
      <td>0.023229</td>
      <td>1.499344e-07</td>
    </tr>
  </tbody>
</table>
<p>373 rows × 6 columns</p>
</div>




```python
# Plot of fault types across different locations and ids
fig, ax1 = plt.subplots(figsize=(7,7))
ax1.scatter(y= final_result.loc[final_result.predicted==0,'probability 0'],x= final_result.loc[final_result.predicted==0,'location'],alpha=0.8,color='g',label = 'fault_severity 0',s= 16)
ax1.scatter(y= final_result.loc[final_result.predicted==1,'probability 1'],x = final_result.loc[final_result.predicted==1,'location'],alpha=0.8,color='r',label = 'fault_severity 1',s= 16)
ax1.scatter(y = final_result.loc[final_result.predicted==2,'probability 2'],x = final_result.loc[final_result.predicted==2,'location'],alpha=0.8,color='b',label = 'fault_severity 2',s= 16)
ax1.set_ylim((0,1))
ax1.set_xlim((0,1200))
ax1.set_ylabel('Probability')
ax1.set_xlabel('location')
ax1.legend(bbox_to_anchor=(1.1, 0.5), loc='upper left')
```




    <matplotlib.legend.Legend at 0x2362e44e908>




![png](output_82_1.png)



```python
final_result.to_csv('Service_Prediction_Result.csv')
```


```python

```

![Screenshot](https://raw.githubusercontent.com/sergiokopplin/indigo/gh-pages/assets/screen-shot.png)

Example of project - Indigo Minimalist Jekyll Template - [Demo](http://sergiokopplin.github.io/indigo/). This is a simple and minimalist template for Jekyll for those who likes to eat noodles.

---

What has inside?

- Gulp
- BrowserSync
- Stylus
- SVG
- Travis
- No JS
- [98/100](https://developers.google.com/speed/pagespeed/insights/?url=http%3A%2F%2Fsergiokopplin.github.io%2Findigo%2F)

---

[Check it out](http://sergiokopplin.github.io/indigo/) here.
If you need some help, just [tell me](http://github.com/sergiokopplin/indigo/issues).
