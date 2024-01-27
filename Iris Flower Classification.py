#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')
#c1=cm.get_cmap('Paired')
c1=cm.get_cmap('jet')
from sklearn.preprocessing import LabelEncoder


# In[4]:


data=pd.read_csv('Iris.csv')


# ## Data Cleaning :- 

# In[5]:


data.duplicated().sum()


# In[6]:


data.info()


# In[7]:


data.describe()


# ## Data Transformation :- 

# In[8]:


LE=LabelEncoder()
sol=LE.fit_transform(np.array(data['Species']).reshape(-1,1))
data['Species_fmt']=sol


# ## Data Visualisation :-

# ### Species Frequency 

# In[9]:


sb.countplot(data=data,x='Species',hue='Species',palette='hsv')
plt.ylabel('Frequency')
plt.title('Species Frequency')
plt.legend()
plt.show()


# ### Sepal length vs Sepal width 

# In[10]:


sb.scatterplot(data=data,x='SepalLengthCm',y='SepalWidthCm',hue='Species',palette='Set1')
plt.title('Sepal Length Vs Sepal Width')


# ### Petal length vs petal width 

# In[11]:


sb.scatterplot(data=data,x='PetalLengthCm',y='PetalWidthCm',hue='Species',palette='autumn')
plt.title('Petal Length Vs Petal Width')


# ### Petal width distribution  

# In[12]:


sb.histplot(data['PetalWidthCm'],kde=True,color=c1(0.3))
plt.ylabel('Occurrence')
plt.title('PetalWidthCm Distribution')


# In[13]:


bins=[0,0.5,1,1.5,2,2.5]
val=pd.cut(data['PetalWidthCm'],bins=bins).value_counts()
sb.barplot(x=val.index,y=val,palette='gist_rainbow')
plt.ylabel('Frequency')
plt.title('Petal Width Frequency')
for i,j in enumerate(val.sort_index()):
    plt.text(i,j,j,va='bottom')


# ### Petal length distribution 

# In[14]:


sb.histplot(data['PetalLengthCm'],kde=True,color=c1(0.4))
plt.ylabel('Occurrence')
plt.title('PetalLengthCm Distribution')


# In[15]:


bins=[x for x in range(1,8)]
val=pd.cut(data['PetalLengthCm'],bins=bins).value_counts()
sb.barplot(x=val.index,y=val,palette='gist_rainbow_r')
plt.ylabel('Frequency')
plt.title('Petal Length Frequency')
for i,j in enumerate(val.sort_index()):
    plt.text(i,j,j,va='bottom')


# ### Sepal length distribution 

# In[16]:


sb.histplot(data['SepalLengthCm'],kde=True,color=c1(0.5))
plt.ylabel('Occurrence')
plt.title('SepalLengthCm Distribution')


# In[17]:


bins=[x for x in np.arange(4,9,0.5)]
val=pd.cut(data['SepalLengthCm'],bins=bins).value_counts()
sb.barplot(x=val.index,y=val,palette='gist_rainbow')
plt.ylabel('Frequency')

for i,j in enumerate(val.sort_index()):
    plt.text(i,j,j,va='bottom')
plt.xticks(rotation=65)
plt.title('Sepal Length Frequency')


# ### Sepal width distribution 

# In[18]:


sb.histplot(data['SepalWidthCm'],kde=True,color=c1(0.7))
plt.ylabel('Occurrence')
plt.title('SepalWidthCm Distribution')


# In[19]:


bins=[2,2.5,3,3.5,4,4.5]
val=pd.cut(data['SepalWidthCm'],bins=bins).value_counts()
sb.barplot(x=val.index,y=val,palette='hsv')
plt.ylabel('Frequency')
plt.title('Sepal Width Frequency')
for i,j in enumerate(val.sort_index()):
    plt.text(i,j,j,va='bottom')


# ## Heatmap for correlation :- 

# In[20]:


sb.heatmap(data=data.drop(['Species','Id'],axis=1).corr(),annot=True,cmap='copper')


# ### Pair plot 

# In[21]:


sb.pairplot(data=data)


# In[ ]:




