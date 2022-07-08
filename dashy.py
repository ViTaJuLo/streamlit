#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 21:18:39 2022

@author: vivian
"""

import streamlit as st


def main():
    
    #st.sidebar.markdown("# Home 🛖")
    #st.image("/Users/vivian/Desktop/edeka.jpeg", use_column_width=False, width= 100)
    st.title("Customer Satisfaction Dashboard")
    st.subheader("Wie schneidet Edeka Kohler im Vergleich zu anderen Lebensmittelketten ab?" )
    st.caption ("basierend auf Google Reviews")
    
if __name__ == '__main__':
    main()
    
    



    
#selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
#page_names_to_funcs[selected_page]()
import streamlit as st
import numpy as np
#import plotly.express as px
import pandas as pd
import datetime
#import sklearn
#import matplotlib.pyplot as plt
#import altair as alt
# 
# 
url1 = 'https://raw.githubusercontent.com/ViTaJuLo/streamlit/main/df'

# ### MAIN PAGE #######
# # Title & subtitle for main page
#st.image("/Users/vivian/Desktop/edeka.jpeg", use_column_width=False, width= 100)
#st.title("Customer Satisfaction Dashboard")
#st.subheader("Wie schneidet Edeka im Vergleich zu anderen Lebensmittelketten ab?" )
#st.caption ("basierend auf Google Reviews")

import base64 
# 
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
# 
# 
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
# 
#set_background('/Users/vivian/Desktop/22.jpg')
# 
st.markdown(f'<h1 style="color:#808080;font-size:24px;">{"_______________________________________________________________________________________________________"}</h1>', unsafe_allow_html=True)
st.subheader("Wozu dient das Dashboard?")
body = "Dieses Dashboard ermöglicht es Insights aus den Google Reviews von Edeka Kohler und der Konkurrenz zu generieren. Als Datenquelle dienen dabei 62 Stores, die mit Data Science Techniken auf ihre numerischen und linguistischen Informationen analysiert wurden."
st.markdown(body, unsafe_allow_html=False)

st.markdown(f'<h1 style="color:#808080;font-size:24px;">{"_______________________________________________________________________________________________________"}</h1>', unsafe_allow_html=True)


    
st.subheader("Wies ist das Dashboard zu nutzen?")
body1 = "Auf der linken Seite befindet sich ein Menü, worüber die beiden Dashboards zu lingustischen oder numerischen Analysen aufgerufen werden können. Ist eine der beiden Unterseiten angewählt, können ein oder mehrere Stores ausgesucht und verglichen werden."
st.markdown(body1, unsafe_allow_html=False)

st.markdown(f'<h1 style="color:#808080;font-size:24px;">{"_______________________________________________________________________________________________________"}</h1>', unsafe_allow_html=True)
st.subheader("Wie sehen die Datenquellen aus?")
body2 = "Es werden zwei Datensätze verwendet. Der erste Datensatz beinhaltet alle Ratings, auch diejenigen, die keine schriftlichen Reviews besitzen. Der zweite Datensatz beinhaltet lediglich diejnigen Ratings, die ebenfalls eine schriftliche Bewertung beinhalten. Die Datensätze werden unten angezeigt."
st.markdown(body2, unsafe_allow_html=False)


@st.cache
# # function to read in store data
def read_df1():
    
    df = pd.read_csv(url1, sep=',')
    return df

df = read_df1()
st.write(df)


### SIDEBAR ####


st.sidebar.subheader("Auswahl der Stores")
# 
tickers = df['new_place_id'].unique()
dropdown = st.sidebar.multiselect('Welche Stores möchten Sie vergleichen?', tickers)
# 
filtered_df = df[df["new_place_id"].isin(dropdown)]

#### POSITIV NEGATIV

# deriving positive and negative frames
negative = filtered_df.query("polarity < 0")
st.write(negative)

#print(len(negative["cleaned_review"]))
positive = filtered_df.query("polarity >= 0")
st.write(positive)

#### 
#df['review_datetime_utc'] = pd.to_datetime(df['review_datetime_utc'])
#df['review_datetime_utc']=df['review_datetime_utc'].dt.date
st.slider("Select the datime!",value=filtered_df.query("review_datetime_utc == '2022-02-01'")
#st.slider("Select the datime!",value=(filtered_df["review_datetime_utc"][1],df["review_datetime_utc"][len(df)-1]))
# defining bi & trigrams
