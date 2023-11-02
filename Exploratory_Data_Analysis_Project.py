#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import dtale
import sweetviz as sv
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab


# In[2]:


# set encoding to latin-1 instead of utf-8 so that excel file can be read
data = pd.read_csv("C:/Users/cbrun/Documents/portfolio_project_2/spotify-2023.csv", encoding="latin-1")
data.head(5)


# In[71]:


data.info()


# In[73]:


data.duplicated().values


# In[74]:


# non of the indices are duplicated
data.duplicated().sum()


# ## Checking for categorical variables

# In[278]:


data['track_name'].value_counts()


# In[19]:


data['artist(s)_name'].value_counts()


# In[20]:


data['key'].value_counts()


# In[21]:


data['mode'].value_counts()


# In[22]:


data['artist_count'].value_counts()


# In[3]:


ProfileReport(data)


# ## Removing text data from dataframe
# This is for the purpose of data preprocessing and not for building a predictive model

# In[3]:


# i am removing the text or string data from the dataset because it will slow down the training process and negatively impact
# the model accuracy
data = data.drop(columns=['track_name', 'artist(s)_name', 'streams', 'in_deezer_playlists', 'in_shazam_charts'])
data


# ## Missing data points for key feature (random variable)

# In[177]:


count = 0
for idx, value in enumerate(pd.isnull(data['key'])):
    if (value):
        count += 1
        print(idx, value, count)


# In[178]:


# confirming that there is 95 missing values for the key feature
data['key'].isna().sum()


# In[37]:


# handling missing values using missing value ratio technique (mvr)
# if mvr is more than the avarage mvr then remove variable
mvr_dataframe = {'Variables': ['in_spotify_playlists', 'in_spotify_charts', 'in_deezer_charts', 'in_apple_playlists', 'in_apple_charts',
                          'danceability_%', 'artist_count', 'valence_%', 'mode', 'key', 'speechiness_%', 'energy_%', 'released_year', 'released_month',
                     'released_day', 'bpm', 'liveness_%', 'acousticness_%', 'instrumentalness_%'],
        'Missing Value Ratio': [f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}",
       f"{0/953:.0%}", f"{0/953:.0%}", f"{95/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}", f"{0/953:.0%}"]}
pd.DataFrame(mvr_dataframe, index=np.arange(19))


# In[27]:


# if mvr is less than 10% of the entire dataset then ignore missing values
mvr_1 = (95/953)*100
key_data = 953 * .10
print(mvr_1 < key_data)


# Since the missing values are less than the proposed threshold 10% < [(10/19)*100] and less than 10% of the entire dataset, I will chose not to bother them. Normally i would monitor them through the entire project and through predictive analysis to see how they effect predictive analysis, however since the project is just for exploratory data analysis, i will leave the missing values alone

# ## Checking distribution of random variables

# In[57]:


#drop categorical variables and check distribution of numerical variables
numerical_data = data.copy()
numerical_data = numerical_data.drop(columns=['artist_count','mode', 'key', 'released_day', 'released_month', 'released_year'])
numerical_data.columns


# In[78]:


numerical_data.skew()


# - The variables with skewness > 1 are price are highly positively skewed.
# - The variables with skewness < -1 are highly negatively skewed.
# - The variables with 0.5 < skewness < 1 are moderately positively skewed.
# - The variables with -0.5 < skewness < -1 are moderately negatively skewed.
# - the variables with -0.5 < skewness < 0.5 are symmetric i.e normally distributed 

# ### visualizations
# - histogram w/ kde curve
# - Q-Q plot
# - boxplot

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, axes = plt.subplots(8,2, figsize=(15,25))
fig.delaxes(axes[0,0])
fig.delaxes(axes[0,1])
fig.delaxes(axes[1,0])
#sns.histplot(numerical_data['released_year'], line_kws={'color': 'red'}, bins=10, ax=axes[0,0])
#sns.histplot(numerical_data['released_month'], line_kws={'color': 'red'}, bins=12, ax=axes[0,1])
#sns.histplot(numerical_data['released_day'], line_kws={'color': 'red'}, bins=20, ax=axes[1,0])
sns.histplot(numerical_data['in_spotify_playlists'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[1,1])
sns.histplot(numerical_data['in_spotify_charts'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[2,0])
sns.histplot(numerical_data['in_apple_charts'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[2,1])
sns.histplot(numerical_data['in_apple_playlists'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[3,0])
sns.histplot(numerical_data['in_deezer_charts'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[3,1])
sns.histplot(numerical_data['danceability_%'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[4,0])
sns.histplot(numerical_data['valence_%'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[4,1])
sns.histplot(numerical_data['speechiness_%'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[5,0])
sns.histplot(numerical_data['energy_%'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[5,1])
sns.histplot(numerical_data['bpm'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[6,0])
sns.histplot(numerical_data['liveness_%'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[6,1])
sns.histplot(numerical_data['acousticness_%'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[7,0])
sns.histplot(numerical_data['instrumentalness_%'], kde='True', line_kws={'color':'red'}, bins=20, ax=axes[7,1])
plt.show()


# In[17]:


def normality(feature, pic_title_1, pic_title_2):
    plt.figure(figsize=(10,5))
    #ax = fig.add_subplot(1,2,1)
    plt.subplot(1,2,1)
    sns.histplot(numerical_data[feature], kde= 'True', line_kws={'color': 'red'}, bins=10)
    plt.title(pic_title_1)
    plt.subplot(1,2,2)
    stats.probplot(numerical_data[feature], plot=pylab)
    plt.title(pic_title_2)


# In[21]:


#normality('released_year', 'released_year histogram', 'released_year prob plot'), 
normality('in_spotify_playlists', 'in_spotify_playlists histogram', 'in_spotify_playlists prob plot'), 
normality('in_spotify_charts', 'in_spotify_charts histogram', 'in_spotify_charts prob plot'), 
normality('in_apple_charts', 'in_apple_charts histogram', 'in_apple_charts prob plot'), 
normality('in_apple_playlists', 'in_apple_playlists histogram', 'in_apple_playlists prob plot'), 
normality('in_deezer_charts', 'in_deezer_charts histogram', 'in_deezer_charts prob plot'), 
normality('danceability_%', 'danceability_% histogram', 'danceability_% prob plot'),
normality('valence_%', 'valence_% histogram', 'valence_% prob plot'), 
normality('speechiness_%', 'speechiness_% histogram', 'speechiness_% prob plot'), 
normality('energy_%', 'energy_% histogram', 'energy_% prob plot'), 
normality('bpm', 'bpm histogram', 'bpm prob plot'), 
normality('liveness_%', 'liveness_% histogram', 'liveness_% prob plot'),
normality('acousticness_%', 'acousticness_% histogram', 'acousticness_% prob plot'), 
normality('instrumentalness_%', 'instrumentalness_% histogram', 'instrumentalness_% prob plot')


# In[6]:


fig, axes = plt.subplots(8,2, figsize=(15,25))
fig.delaxes(axes[6,1])
fig.delaxes(axes[7,0])
fig.delaxes(axes[7,1])
sns.boxplot(data=numerical_data, x='in_spotify_playlists', ax=axes[0, 0])
sns.boxplot(data=numerical_data, x='in_spotify_charts', ax=axes[0,1])
sns.boxplot(data=numerical_data, x='in_apple_charts', ax=axes[1,0])
sns.boxplot(data=numerical_data, x='in_apple_playlists', ax=axes[1,1])
sns.boxplot(data=numerical_data, x='in_deezer_charts', ax=axes[2,0])
sns.boxplot(data=numerical_data, x='danceability_%', ax=axes[2,1])
sns.boxplot(data=numerical_data, x='valence_%', ax=axes[3,0])
sns.boxplot(data=numerical_data, x='speechiness_%', ax=axes[3,1])
sns.boxplot(data=numerical_data, x='energy_%', ax=axes[4,0])
sns.boxplot(data=numerical_data, x='bpm', ax=axes[4,1])
sns.boxplot(data=numerical_data, x='liveness_%', ax=axes[5,0])
sns.boxplot(data=numerical_data, x='acousticness_%', ax=axes[5,1])
sns.boxplot(data=numerical_data, x='instrumentalness_%', ax=axes[6,0])


# ## BPM Over Time (Per Year) for Bad Bunny

# In[24]:


bad_bunny_data = data.copy()
bad_bunny_data = bad_bunny_data.where(data["artist(s)_name"] == "Bad Bunny")
bad_bunny_data = bad_bunny_data.dropna()


# In[30]:


fig, ax = plt.subplots(figsize=(15, 5))
bpm_years_line_plot = sns.lineplot(data=bad_bunny_data, y="bpm", x="released_year", ax=ax)
bpm_years_line_plot.set_title('Change in bpm over time for ', fontsize = 15, loc="left")
bpm_years_line_plot.set_xlabel('Released Year', fontsize=12, loc="left")
_ = bpm_years_line_plot.set_ylabel('BPM', fontsize=12, loc="bottom")
sns.despine()


# As can be seen, the bpm in bad bunny's released tracks increased drastically from 2020 to 2021 but decreased at about the same rate from 
# 2021 to 2022. The bpm in his released tracks increased again, but at slower rate, from 2022 to 2023. 

# ## Checking Outliers for Random Variables

# In[7]:


# finding outliers of this random variable
def outliers(feature, threshold):
    data = numerical_data[feature]
    z_scores = np.abs(stats.zscore(data)) # applying zscore method
    #threshold = 1.96 #since 95% of the data should be within 1.96 standard deviations below and above the mean 
    outliers = data[z_scores > threshold]
    outliers = np.sort(outliers)
    print(outliers)


# ## in_spotify_playlists outliers

# In[16]:


# feature values
outliers('in_spotify_playlists', 1.96)


# In[23]:


# index values
np.where(numerical_data[['in_spotify_playlists']] >= 20763)


# ## in_spotify_charts outliers

# In[24]:


# feature values
outliers('in_spotify_charts', 1.96)


# In[25]:


# index values
np.where(numerical_data[['in_spotify_charts']] >= 52)


# ## in_apple_charts outliers

# In[26]:


# feature values
outliers('in_apple_charts', 1.96)


# In[27]:


# index values
np.where(numerical_data[['in_apple_charts']] >= 152)


# ## in_apple_playlists outliers

# In[28]:


#feature values
outliers('in_apple_playlists', 1.96)


# In[29]:


# index values
np.where(numerical_data[['in_apple_playlists']] >= 238)


# ## in_deezer_charts outliers

# In[30]:


outliers('in_deezer_charts', 1.96)


# In[31]:


np.where(numerical_data[['in_deezer_charts']] >= 15)


# ## danceability_% outliers

# In[32]:


outliers('danceability_%', 1.96)


# In[54]:


np.sort(numerical_data['danceability_%'].values)


# In[49]:


lower_quartile = int(np.percentile(numerical_data['danceability_%'], 25))
upper_quartile = int(np.percentile(numerical_data['danceability_%'], 75))
x = np.sort(numerical_data['danceability_%'].values)


# In[43]:


x > upper_quartile + 1.5 * (upper_quartile-lower_quartile)


# In[46]:


x < lower_quartile - 1.5 * (upper_quartile-lower_quartile)


# ## valence_% outliers

# In[47]:


outliers('valence_%', 1.96)


# In[51]:


np.where(numerical_data['valence_%'] <= 5)


# ## speechiness_% outliers

# In[57]:


outliers('speechiness_%', 1.96)


# In[63]:


lower_quartile = int(np.percentile(numerical_data['speechiness_%'], 25))
upper_quartile = int(np.percentile(numerical_data['speechiness_%'], 75))
x = np.sort(numerical_data['speechiness_%'].values)


# In[67]:


x > upper_quartile + 1.5 * (upper_quartile-lower_quartile)


# ## energy_% outliers

# In[68]:


outliers('energy_%', 1.96)


# In[70]:


np.sort(numerical_data['energy_%'])


# # bpm outliers

# In[8]:


outliers('bpm', 1.96)


# # liveness_%

# In[9]:


outliers('liveness_%', 1.96)


# ## acousticness_%

# In[11]:


outliers('acousticness_%', 1.96)


# ## instrumentalness_%

# In[12]:


outliers('instrumentalness_%', 1.96)


# ## Data Transformation by Normalization (Using One Random Variable or Feature)
# Transforming the data will create a more gaussian-like distribution or bell-like curve. Normalizing the data will help model prediction accuracy since models run under the assumption that the data is normally distributed. 
# - I won't be building a machine learning pipeline since the goal is not to actually build a model but to simply prepare the data for modeling
# - Before removing outliers we first must normalize the data

# ## Log Transformation Function

# In[58]:


inspotplay_data = numerical_data['in_spotify_playlists'].copy()


# In[38]:


log_transformed_inspotplay_data = np.log(inspotplay_data)


# In[47]:


def log_tranformation_viz(pic_title_1, pic_title_2):
    plt.figure(figsize=(10,5))
    #ax = fig.add_subplot(1,2,1)
    plt.subplot(1,2,1)
    sns.histplot(log_transformed_inspotplay_data, kde= 'True', line_kws={'color': 'red'}, bins=10)
    plt.title(pic_title_1)
    plt.subplot(1,2,2)
    stats.probplot(log_transformed_inspotplay_data, plot=pylab)
    plt.title(pic_title_2)


# In[48]:


log_tranformation_viz('insp histogram', 'insp qq plot')


# ## Reciprocal Tranformation Function

# In[59]:


reciprocal_transformed_inspotplay_data = 1/inspotplay_data


# In[60]:


def reciprocal_tranformation_viz(pic_title_1, pic_title_2):
    plt.figure(figsize=(10,5))
    #ax = fig.add_subplot(1,2,1)
    plt.subplot(1,2,1)
    sns.histplot(reciprocal_transformed_inspotplay_data, kde= 'True', line_kws={'color': 'red'}, bins=10)
    plt.title(pic_title_1)
    plt.subplot(1,2,2)
    stats.probplot(reciprocal_transformed_inspotplay_data, plot=pylab)
    plt.title(pic_title_2)


# In[61]:


reciprocal_tranformation_viz('insp histogram', 'insp qq plot')


# ## Sqrt Tranformed Function

# In[62]:


sqrt_transformed_inspotplay_data = np.sqrt(inspotplay_data)


# In[67]:


def sqrt_transformation_viz(pic_title_1, pic_title_2):
    plt.figure(figsize=(10,5))
    #ax = fig.add_subplot(1,2,1)
    plt.subplot(1,2,1)
    sns.histplot(sqrt_transformed_inspotplay_data, kde= 'True', line_kws={'color': 'red'}, bins=10)
    plt.title(pic_title_1)
    plt.subplot(1,2,2)
    stats.probplot(sqrt_transformed_inspotplay_data, plot=pylab)
    plt.title(pic_title_2)


# In[68]:


sqrt_transformation_viz('sqrt histogram', 'sqrt qq plot')


# Log Transformation accurately transforms the chosen variable into a normally distributed variable
