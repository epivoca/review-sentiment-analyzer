import streamlit as st
import nltk
import mtranslate
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class TextSentimentAnalyzer:
    @staticmethod
    def analyze_sentiment(text: str) -> float:
        sia = SentimentIntensityAnalyzer()

        score = sia.polarity_scores(text)

        return score["compound"]


st.title("Simple Sentiment Analysis")
st.write("Enter a text message below to analyze its sentiment:")

text_input = st.text_area("Enter text here:")

if st.button("Analyze Sentiment"):
    if text_input:
        en_text = mtranslate.translate(from_language='ru', to_language='en', to_translate=text_input)

        compound_score = TextSentimentAnalyzer.analyze_sentiment(en_text)

        if compound_score >= 0.05:
            st.write("Sentiment: Positive :smile:")
        elif -0.05 < compound_score < 0.05:
            st.write("Sentiment: Neutral :neutral_face:")
        else:
            st.write("Sentiment: Negative :disappointed:")

        st.write(f"Sentiment Score (Compound): {compound_score}")
    else:
        st.warning("Please enter a text message to analyze.")
