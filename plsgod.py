#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 18:25:39 2022

@author: vivian
"""

import streamlit as st

# LOGO
st.image("https://raw.githubusercontent.com/ViTaJuLo/streamlit/main/edeka.jpeg", use_column_width=False, width= 100)



# HEAD AND DESCRIPTION OF PAGE 
st.title("Customer Satisfaction Dashboard")
st.subheader("Wie schneidet Edeka Kohler im Vergleich zu anderen Lebensmittelketten ab?" )
st.caption ("basierend auf Google Reviews")




# BACKGROUND OF APP
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://raw.githubusercontent.com/ViTaJuLo/streamlit/main/22.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack_url()





# IMPORTING NECESSARY LIBRARIES
import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import datetime
import sklearn
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns






# IMPORTING AND SETTING UP DATA
url1 = 'https://raw.githubusercontent.com/ViTaJuLo/streamlit/main/df'

@st.cache
#function to read in store data
def read_df1():
    df = pd.read_csv(url1, sep=',')
    return df

# read in data
df = read_df1()

## FIRST VISUALS
st.markdown(f'<h1 style="color:#808080;font-size:24px;">{"_______________________________________________________________________________________________________"}</h1>', unsafe_allow_html=True)
st.subheader("Der Vergleich: Edeka vs. Andere")
body = "Die unten abgebildete Matrix erlaubt es verschiedene Standorte mit Hinblick auf diverse Key Metriken zu vergleichen"
st.markdown(body, unsafe_allow_html=False)

#start = df["review_datetime_utc"][0]
#end = df["review_datetime_utc"][len(df)-1]


# CREATE DATA FILTERS 
tickers = df['new_place_id'].unique()
dropdown = st.multiselect('Welche Standorte möchten Sie vergleichen?', tickers, default=["Edeka Kohler Kehl  - Am Läger"])
filtered_df = df[df["new_place_id"].isin(dropdown)]
#st.slider("Select the datime!",value=(start, end))
st.write(filtered_df)


group = filtered_df.groupby(['new_place_id'], as_index=False).agg({'polarity_reviews': ['mean'], 'review_rating': ['mean','count']})
pivot = group.pivot_table(columns="new_place_id")
fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='blackbody')
fig.show()
st.plotly_chart(fig, use_container_width=True)


### CATEGORIES ####
st.subheader("Wie wurden die Kategorien Preis-Leistungsverhältnis, Service, Produkte & Innendesign von Kunden bewertet?")
body = "Diese Matrix visualisiert pro Kategorie den durchschnittlichen Sentiment-Score für die ausgewählten Geschäfte und den ausgewählten Zeitraumen. Der Score bewegt sich zwischen -1 und 1, wobei -1 sehr negativ ist und +1 sehr positiv."
st.markdown(body, unsafe_allow_html=False)

st.set_option('deprecation.showPyplotGlobalUse', False)
preis_df = filtered_df.query("preis == 1")
preis_df['preis'] = np.where(preis_df['preis']== 1, "preis", 0)
preis_group = preis_df.groupby(['new_place_id', "preis"], as_index=False)['polarity'].mean()
pivot_preis = preis_group.pivot_table('polarity', index='new_place_id', columns=('preis'))
#fig = sns.set(rc={'figure.figsize':(11.7,8.27)})
#swarm_plot = sns.heatmap(pivot_preis, cmap="vlag_r", annot=True, cbar=False, annot_kws = {'fontsize': 10 }, linewidth = 1)


service_df = filtered_df.query("service == 1")
service_df['service'] = np.where(service_df['service']== 1, "service", 0)
service_group = service_df.groupby(['new_place_id', "service"], as_index=False)['polarity'].mean()
pivot_service = service_group.pivot_table('polarity', index='new_place_id', columns=('service'))
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#swarm_plot = sns.heatmap(pivot_service, cmap="vlag_r", annot=True, cbar=False, annot_kws = {'fontsize': 10 }, linewidth = 1)


produkt_df = filtered_df.query("produkte == 1")
produkt_df['produkte'] = np.where(produkt_df['produkte']== 1, "produkte", 0)
produkt_group = produkt_df.groupby(['new_place_id', "produkte"], as_index=False)['polarity'].mean()
pivot_produkt = produkt_group.pivot_table('polarity', index='new_place_id', columns=('produkte'))
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#swarm_plot = sns.heatmap(pivot_produkt, cmap="vlag_r", annot=True, cbar=False, annot_kws = {'fontsize': 10 }, linewidth = 1)


design_df = filtered_df.query("innendesign == 1")
design_df['innendesign'] = np.where(design_df['innendesign']== 1, "design", 0)
design_group = design_df.groupby(['new_place_id', "innendesign"], as_index=False)['polarity'].mean()
pivot_design = design_group.pivot_table('polarity', index='new_place_id', columns=('innendesign'))
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#swarm_plot = sns.heatmap(pivot_design, cmap="vlag_r", annot=True, cbar=False, annot_kws = {'fontsize': 10 }, linewidth = 1)

frames = [pivot_preis, pivot_service, pivot_produkt, pivot_design]

result = pd.concat(frames, axis=1)
#fig = sns.set(rc={'figure.figsize':(11.7,8.27)})
#fig = sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
#swarm_plot = sns.heatmap(result, cmap="vlag_r", annot=True, cbar=False, annot_kws = {'fontsize': 10 }, linewidth = 1)
#plt.show()
#st.pyplot(fig, use_container_width=True)
fig = px.imshow(result, text_auto=True, aspect="auto", color_continuous_scale='purples')
fig.update_layout(coloraxis_showscale=False)
fig.show()
st.plotly_chart(fig, use_container_width=True)







#####  SECOND VISUALS: RATINGS OVER TIME ####


st.subheader("Entwicklung der Ratings über Zeit")
body = "Diese Grafik visualisiert das durchschnittliche Rating per Standort über Zeit"
st.markdown(body, unsafe_allow_html=False)
group = filtered_df.groupby(['new_place_id', 'Review_year'], as_index=False)['review_rating'].mean()
fig = px.line(group, x=group['Review_year'], y=group['review_rating'], width=1000, height=400, color=group['new_place_id'], title='Entwicklung des durschnittlichen Ratings pro Jahr pro Standort')
#fig.show()
fig = fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
st.plotly_chart(fig, use_container_width=True)


st.markdown(f'<h1 style="color:#808080;font-size:24px;">{"_______________________________________________________________________________________________________"}</h1>', unsafe_allow_html=True)








# POSITIVE N-GRAMS 
st.subheader("Positives Feedback per Standort")
body = "Diese Grafik zeigt das positive Feedback anhand derjenigen Reviews, die ein Sentiment größer oder gleich 0 aufweisen"
st.markdown(body, unsafe_allow_html=False)
positive = df.query("polarity >= 0")
tickers2 = positive['new_place_id'].unique()
dropdown2 = st.multiselect('Welchen Standort möchten Sie auswählen?', tickers2, default=["Edeka Kohler Kehl  - Am Läger"])

pos_df = positive[positive["new_place_id"].isin(dropdown2)]
from sklearn.feature_extraction.text import CountVectorizer

c_vec = CountVectorizer(ngram_range=(2,3)) 
ngrams = c_vec.fit_transform(pos_df['cleaned_review'].apply(lambda x: np.str_(x)))
count_values = ngrams.toarray().sum(axis=0)
vocab = c_vec.vocabulary_
positive_ngrams = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)).rename(columns={0: 'frequency', 1:'ngram'})
st.write(positive_ngrams)


from wordcloud import WordCloud
positive_ngrams_top = positive_ngrams[:10]
fig = px.bar(positive_ngrams_top, x="frequency" , y="ngram", title='Top 10 - Positive N-grams')
fig = fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
fig = fig.update_traces(marker_color='green')
st.plotly_chart(fig, use_container_width=True)
   

st.markdown(f'<h1 style="color:#808080;font-size:24px;">{"_______________________________________________________________________________________________________"}</h1>', unsafe_allow_html=True)












# Negative Ngrams
st.subheader("Negatives Feedback per Standort")
body = "Diese Grafik zeigt das negative Feedback anhand derjenigen Reviews an, die ein Sentiment zwischen -1 und 0 aufweisen. Anmerkung: Leider sagt der Algorithmus nicht alle negativen Feedbacks richtig voraus. Deswegen kann es hier zu Unstimmigkeiten kommen."
st.markdown(body, unsafe_allow_html=False)
negative = df.query("polarity < 0")
tickers3 = negative['new_place_id'].unique()
dropdown3 = st.multiselect('Welchen Standort möchten Sie auswählen?', tickers3, default=["Edeka Kohler Kehl  - Am Läger"])

neg_df = negative[negative["new_place_id"].isin(dropdown3)]
from sklearn.feature_extraction.text import CountVectorizer


c_vec = CountVectorizer(ngram_range=(2,3)) 
ngrams = c_vec.fit_transform(neg_df['cleaned_review'].apply(lambda x: np.str_(x)))
count_values = ngrams.toarray().sum(axis=0)
vocab = c_vec.vocabulary_
negative_ngrams = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)).rename(columns={0: 'frequency', 1:'ngram'})
st.write(negative_ngrams)  

negative_ngrams_top = negative_ngrams[:10]
fig = px.bar(negative_ngrams_top, x="frequency" , y="ngram", title='Top 10 - Negative N-grams')
fig = fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
fig = fig.update_traces(marker_color='blue')
st.plotly_chart(fig, use_container_width=True)
