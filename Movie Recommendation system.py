#!/usr/bin/env python
# coding: utf-8

# # Importing importat lybrari

# In[1]:


import numpy as np
import pandas as pd


# # Importing Data-Set

# In[2]:


movies = pd.read_csv(r'B:\Worked dataset\movie 5000\tmdb_5000_movies.csv')
credits = pd.read_csv(r'B:\Worked dataset\movie 5000\tmdb_5000_credits.csv')
print("shape of movies: ",movies.shape)
print("shape of credits: ",credits.shape)


# # Knowing the Data-set

# # Data set =  Movies

# In[3]:


movies.head(1)


# #### We have this columns in movies data-set

# In[4]:


#budget
#genres
#homepage
#id
#keywords
#original_language
#original_title
#overview
#popularity
#production_companies
#production_countries
#release_date
#revenue
#runtime
#spoken_languages
#status
#tagline
#title#vote_average
#vote_coun


# # Data set = Credits

# In[5]:


credits.head(1)


# #### We have this columns in credits data set

# In[6]:


#movie_id
#title
#cast
#crew


# #### Going deep insight of data to finding logic = cast values, crew values

# In[7]:


credits.head(1)['cast'].values


# In[8]:


credits.head(1)['crew'].values


# # Stages of data processing 
# STEPS =
# 1) merging the both data set on the basis of common column for convenince and working on one data set.
# 
# 2) Removing unwanted columns.
# 
# 3) simplifying the joins (new data set formed) data set.
# 
# 4) 4) Created new data frame that data fram only includes three columns, using merging method.
# 
# 5) Data preprocessing = 
# >* droping null values.
# >* data transformation. = Converting strin into list.
# >* d

# ## 1) merging the data set on the basis of common column  = "title"

# In[9]:


movies.shape


# In[10]:


credits.shape


# #### merg data set

# In[11]:


movies.merge(credits,on = 'title').shape


# #### Reassigning merg data set in movies

# In[12]:


movies = movies.merge(credits,on = 'title')


# In[13]:


movies.head(1)


# # 2) Removing unwanted columns

# In[14]:


movies.info()


# #### Removing

# In[15]:


#budget
#homepage
#original_language
#original_title
#popularity
#production_companies
#runtime
#spoken_languages
#status
#tagline
#vote_average
#vote_count
#movie_id


# In[16]:


movies['original_language'].value_counts()


# In[17]:


movies['original_language'].value_counts()


# In[ ]:





# In[ ]:





# #### Remaining following columns same as it is

# In[18]:


# genres
# id
# keywords
# title
# overview
# cast
# crew


# In[19]:


movies['genres'].values


# #### Reassigning important and required columns in data set as it is = creating new data frame as movies

# In[20]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[21]:


movies.head(1)


# # 4) Created new data frame include only three columne using merging  data set columns 

# In[22]:


movies.head(1)


# # 5) Data pre processing

# ### 1) Preprocessing =  Droping null values

# In[23]:


movies.isnull().sum()


# In[24]:


movies.dropna(inplace = True)
movies.isnull().sum().sum()


# In[25]:


movies.duplicated().sum()


# ### 2) preprocessing(data transformation) = simplifying the data set columns 

# In[26]:


movies.iloc[0].genres


# ##### '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# 
# ### 1) Transforming like = Converting strin into list for simplifying data.
# ### creating convert function
# 
# ##### ['Action',Adventure','Fantasty','Sci-Fi']

# In[27]:


import ast
ast.literal_eval


# In[28]:


def convert(obj):
   L = []
   for i in ast.literal_eval(obj):
       L.append(i['name'])
   return L


# In[29]:


ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[30]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# #### 2) appling convert function for transformation on genres for simplify data set

# In[31]:


movies['genres'].apply(convert)


# In[32]:


movies['genres'] = movies['genres'].apply(convert)
movies.head(1)


# #### 3) appling convert function on keywords  

# In[33]:


movies['keywords'].apply(convert)


# In[34]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head(1)


# #### 4) we want top or first 3 name of cast from film,create a function to get top 3 name from film. 
# #### Simplifying the cast column 

# In[35]:


movies['cast'][0]


# ##### '[{"cast_id": 242, "character": "Jake Sully", "credit_id": "5602a8a7c3a3685532001c9a", "gender": 2, "id": 65731,
# 
# ##### extract this = "name": "Sam Worthington", "order": 0,

# In[36]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


# In[37]:


movies['cast'].apply(convert3)


# In[38]:


movies['cast'] = movies['cast'].apply(convert3)


# In[39]:


movies.head(1)


# #### 5) we want director name because director can be best for recommendation 
# >#### so we are creating function to fetch director name from crew 

# In[40]:


movies['crew'][0]


# In[41]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[42]:


movies['crew'].apply(fetch_director)


# In[43]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[44]:


movies.head(1)


# #### 6) overview column is string and we have to convert it into list 
# > we create lambda function

# In[45]:


movies['overview'][0]


# In[46]:


movies['overview'].apply(lambda x:x.split())


# In[47]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[48]:


movies.head(1)


# #### 7) remove the space from words = 
# >For not concidering the duplicate words and tags as same word because of it system can not be predict aqurate.
# 
# >Sam Worthington SamMendes = sam is different considering tag but we want same intigrate as samWorthington

# In[49]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[50]:


movies.head()


# #### 8) adding all simplifing columns into one tag name column

# In[51]:


movies['tags'] = movies['overview'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[52]:


movies.head()


# #### 9) Creating final data set 

# In[53]:


new_df = movies[['movie_id','title','tags']]


# In[54]:


new_df


# #### 10) Converting tags(list) column into string

# In[55]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[56]:


new_df.head()


# In[57]:


new_df['tags'][0]


# In[58]:


new_df['tags'][1300]


# #### 11) Converting all words of tags in lower case

# In[59]:


new_df['tags'].apply(lambda x:x.lower())


# # This is final new data frame

# In[60]:


new_df.head()


# # calculating the similarity of movies for recommendation system
# > calculate similarity on the basis of tags text becase it is textual data set
# 
# >we calculate similer words using "Vectorization"
# 
# >conver text into vector using bag of words technique and recommend closest vector

# In[61]:


new_df['tags'][0]


# In[62]:


new_df['tags'][1]


# In[63]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words = 'english')


# In[64]:


cv.fit_transform(new_df['tags']).toarray()


# In[65]:


vectors = cv.fit_transform(new_df['tags']).toarray()
vectors.shape


# In[66]:


vectors[0]


# In[67]:


len(cv.get_feature_names())


# In[68]:


cv.get_feature_names()


# # Because of word similarity we have to use "Steming" tecnique 
# >steming apply on dataset

# ### This is steming =
# 
# ['loved','loving','love']
# 
# ['love','love','love']

# In[69]:


import nltk #nltk nature language processig lybrari


# In[70]:


from nltk.stem.porter import PorterStemmer
#Creating object
ps = PorterStemmer()


# In[71]:


ps.stem('loveed')


# In[72]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[73]:


new_df['tags'][0]


# In[74]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d Action Adventure Fantasy ScienceFiction JamesCameron')


# In[75]:


new_df['tags'].apply(stem)


# In[76]:


new_df['tags'] = new_df['tags'].apply(stem)


# ### Because of word similarity we have to use "Steming" tecnique 

# In[77]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words = 'english')


# In[78]:


vectors = cv.fit_transform(new_df['tags']).toarray()
vectors.shape


# In[79]:


vectors


# In[80]:


vectors[0]


# In[81]:


cv.get_feature_names()


# In[ ]:





# In[ ]:





# In[82]:


from sklearn.metrics.pairwise import cosine_similarity


# In[83]:


cosine_similarity(vectors)


# In[84]:


similarity = cosine_similarity(vectors)


# In[85]:


# movie similarity with each movie


# In[86]:


similarity[0]


# In[87]:


sorted(similarity[0])[-10:-1]


# In[88]:


#most similar movies


# In[89]:


sorted(similarity[0],reverse = True)


# In[90]:


sorted(list(enumerate(similarity[0])),reverse = True,key=lambda x:x[1])[1:6]


# # Creating recommend system

# In[91]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(similarity[0])),reverse = True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(i[0])


# In[92]:


recommend("Avatar")


# In[93]:


new_df.iloc[1216]


# In[94]:


new_df.iloc[1216].title


# In[95]:


new_df[new_df['title'] == 'Avatar']


# In[96]:


new_df[new_df['title'] == 'Avatar'].index[0]


# In[97]:


new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[98]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(similarity[0])),reverse = True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[99]:


recommend('Batman Begins')


# In[ ]:





# In[100]:


import pickle 


# In[101]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[102]:


new_df['title'].values


# In[103]:


new_df.to_dict()


# In[104]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[105]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




