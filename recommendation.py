#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


path = "/home/kamrul/Documents/pro/Celluloid/archive/"
credits_df = pd.read_csv(path + "tmdb_5000_credits.csv")
movies_df = pd.read_csv(path + "tmdb_5000_movies.csv")


# In[3]:


credits_df.columns = ['id','title', 'cast','crew']
movies_df = movies_df.merge(credits_df, on="id")


# In[4]:


movies_df['title_x'] = [title.lower() for title in movies_df['title_x']]


# In[5]:


features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)
movies_df[features].head(10)


# In[6]:


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


# In[7]:


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


# In[9]:


movies_df["director"] = movies_df["crew"].apply(get_director)
features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)


# In[10]:


def clean_data(row):
    if isinstance(row, list):
        return [str.lower(i.replace(" ", "")) for i in row]
    else:
        if isinstance(row, str):
            return str.lower(row.replace(" ", ""))
        else:
            return ""
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)


# In[11]:


def create_soup(features):
    return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['director'] + ' ' + ' '.join(features['genres'])
movies_df["soup"] = movies_df.apply(create_soup, axis=1)
print(movies_df["soup"][0])


# In[12]:


count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix) 
movies_df = movies_df.reset_index()


# In[13]:


indices = pd.Series(movies_df.index, index=movies_df["title_x"]).drop_duplicates()


# In[23]:


def get_recommendations(title, cosine_sim):
    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores= sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores= similarity_scores[1:11]
    # (a, b) where a is id of movie, b is similarity_scores
    movies_indices = [ind[0] for ind in similarity_scores]
    movies = movies_df["title_x"].iloc[movies_indices]
    return movies
print("################ Content Based System #############")
print("Recommendations for The Dark Knight Rises")
print(get_recommendations("the dark knight rises", cosine_sim2))
print()
print("Recommendations for Avengers")
print(get_recommendations("the avengers", cosine_sim2))


# In[24]:


count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["title_x"])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix) 


# In[93]:


def get_sim_scores(title):
    count_vectorizer = CountVectorizer()
    data = list(movies_df["title_x"]) 
    data.append(title)
    count_matrix = count_vectorizer.fit_transform(data)
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix) 
    similarity_scores = list(enumerate(cosine_sim2[len(data) - 1]))
    return similarity_scores
    


# In[101]:


def get_recommendations2(title):
    similarity_scores = get_sim_scores(title)
    similarity_scores= sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_score= similarity_scores[1]
    movie_index = similarity_score[0]
    movie = movies_df["title_x"].iloc[movie_index]
    return movie


# In[107]:


print(get_recommendations2("knight dark"))


# In[79]:


title = 'ping'
list(title)


# In[ ]:




