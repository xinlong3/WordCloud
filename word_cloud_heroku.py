import streamlit as st
import numpy as np
from wordcloud import WordCloud
from main import keyword_extraction_text_rank

st.title("Welcome to WordCloud!")
st.text("Put your text here and generate your word cloud!")
user_input = st.text_area("text goes here")
extracted_key_words, frequency_dict = keyword_extraction_text_rank(user_input)
word_cloud = WordCloud(width = 800, height = 800, background_color ='white', min_font_size = 10).generate_from_frequencies(frequency_dict)
st.image(word_cloud)
