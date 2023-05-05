#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import requests
import time
import pickle
from config import *

# Set up Spotify API client

auth_manager = SpotifyClientCredentials(client_id=client_id,
                                         client_secret=client_secret_id)
sp = spotipy.Spotify(auth_manager=auth_manager)


# In[2]:


songs_df = pd.read_csv("songs1.csv")


# In[3]:


# Function to get song features from Spotify API
def get_song_features(title, artist):
    """
    Gets the audio features of a song from the Spotify API
    given its title and artist name.
    """
    results = sp.search(q=f"{title} artist:{artist}", type="track", limit=1)
    if results["tracks"]["total"] > 0:
        track_id = results["tracks"]["items"][0]["id"]
        track_features = sp.audio_features([track_id])[0]
        return track_features
    else:
        return None


# In[10]:


def run_recommender():
    title = input ("Enter song: ")
    artist = input("Enter artist: ")
 
    audio_feat = get_song_features(title, artist)
        
    d = {
        "danceability" : [audio_feat["danceability"]],
        "energy" : [audio_feat["energy"]],
        "loudness": [audio_feat["loudness"]],
        "speechiness" : [audio_feat["speechiness"]],
        "acousticness" : [audio_feat["acousticness"]],
        "instrumentalness":  [audio_feat["instrumentalness"]],
        "liveness" : [audio_feat["liveness"]],
        "valence"  : [audio_feat["valence"]],
        "tempo" : [audio_feat["tempo"]],
        "key" : [audio_feat["key"]]

    }
    
    with open("scaler.pickle", "rb") as f:
        stdScaler = pickle.load(f)
        
    feat_scaled = stdScaler.transform(pd.DataFrame(d))
    feat_scaled = pd.DataFrame(feat_scaled, columns = d.keys())

    with open("kmeans_7.pickle", "rb") as f:
        kmeans = pickle.load(f)
        
    cluster = kmeans.predict(feat_scaled)[0]
    print(f"Cluster {cluster}")
    
    res = songs_df[(songs_df["artists"] == artist) &                   (songs_df["title"] == title)]
    if not res.empty:
        # Title/Artist already exists in our songDB
        hot_or_not = res["dataset"]
        c = res["cluster2"]
        
        rec_song = songs_df[(songs_df["dataset"] == hot_or_not) &                  (songs_df["cluster2"] == c)].sample()[["title", "artists", "id"]]
    else:
        rec_song = songs_df[(songs_df["cluster2"] == cluster) &                  (songs_df["dataset"] == "not_hot_songs")].sample()[["title", "artists", "id"]]
        
    print(rec_song)


# In[11]:


def more_recommendations():
    while True:
        run_recommender()
    
        quit = input("Wanne quit? (y/n) ")
        if quit == "y":
            break

    print("Good bye!")


# In[12]:


more_recommendations()

