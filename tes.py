import streamlit as st
import pandas as pd
import joblib
import cleantext
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
model = joblib.load('random_forest_sentiment_model.pkl')

# Download necessary NLTK data
nltk.download('stopwords')

# Streamlit Header
st.header('Sentiment Analysis of BSI Mobile User by Reni Wahyuni')

# Text Analysis Section
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        cleaned_text = cleantext.clean(text, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)
        sentiment = model.predict([cleaned_text])[0]
        st.write('Sentiment: ', sentiment)

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))

# CSV/XLSX Analysis Section
with st.expander('Analyze CSV/XLSX'):
    upl = st.file_uploader('Upload file')

    if upl:
        df = pd.read_excel(upl)

        # Data Cleaning
        patterns = [
            r'https\S+',         # URLs
            r'@\S+',             # User Mentions
            r'#\S+',             # Hashtags
            r"'\w+",             # Single Quotes Followed 
            r'[^\w\s]',          # Non-Alphanumeric Characters
            r'\s+',              # Extra Spaces
            r'.\s+',             # Period
            r',\s+',             # Comma
            r'?\s+',             # Question Mark
            r'!\s+',             # Exclamation Mark
        ]
        combined_pattern = '|'.join(patterns)
        df['output_cleaning_casefolding'] = df['content'].str.replace(combined_pattern, ' ', flags=re.IGNORECASE)

        # Case Folding
        df['output_cleaning_casefolding'] = df['output_cleaning_casefolding'].str.lower().str.strip()

        # Tokenizing
        regexp = RegexpTokenizer('\w+')
        df['output_tokenizing'] = df['output_cleaning_casefolding'].apply(regexp.tokenize)

        # Stopwords Filtering
        english_stopwords = set(stopwords.words('english'))
        indonesian_stopwords = set(stopwords.words("indonesian"))
        my_stopwords = ['aplikasi', 'bsi mobile', 'transaksi', 'buka', 'mobile', 'bank', 'habis', 'aktivasi', 'aja', 'saldo',
                        'masuk', 'pakai', 'banget', 'bsi', 'aplikasinya', 'nya', 'this', 'i', 't', 'can', 'bni', 'fiturnya', 'deh']
        indonesian_stopwords.update(my_stopwords)
        all_stopwords = indonesian_stopwords.union(english_stopwords)

        df['output_filtering'] = df['output_tokenizing'].apply(lambda x: [item for item in x if item not in all_stopwords])
        df['final_output_preprocessing'] = df['output_filtering'].apply(lambda x: ' '.join([item for item in x if len(item) > 3]))

        # Predict sentiment using the model
        df['sentiment'] = model.predict(df['final_output_preprocessing'])

        # Display processed data
        st.write(df)

        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Hasil Preprocessing.csv',
            mime='text/csv',
        )

        # Sentiment pie chart
        st.subheader('Sentiment Pie Chart')
        df_sentiment = df['sentiment'].value_counts()
        fig_sentiment = plt.figure(figsize=(8, 8))
        color_sentiment = ['#045D5D', '#FDD017', '#48a39e']
        explode_sentiment = (0.1, 0.1, 0.1)
        df_sentiment.plot(kind='pie', autopct='%1.1f%%', shadow=False, colors=color_sentiment,
                         startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'black'},
                         explode=explode_sentiment, labels=None, pctdistance=0.85,
                         textprops={'color': 'white'})  # Set text color to white
        plt.title('Sentiment Distribution')
        plt.legend(labels=['Positive', 'Negative', 'Neutral'], loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig_sentiment)

        # Rating distribution
        st.subheader('Rating Distribution')
        yearwise_rating = df.groupby('year')['rating'].value_counts().reset_index(name='count').sort_values(by=['year'], ascending=True)
        fig_rating = plt.figure(figsize=(15, 7))
        palette_rating = ['#045D5D', '#FDD017', '#48a39e']
        ax_rating = sns.barplot(x='rating', y='count', hue='year', data=yearwise_rating, palette=palette_rating)
        ax_rating.grid(False)
        for container in ax_rating.containers:
            ax_rating.bar_label(container, fmt='%d', fontsize=10, color='black', label_type='edge', padding=3)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Quantity')
        plt.legend(title='Year')
        st.pyplot(fig_rating)

        # Month-wise sentiment analysis
        st.subheader('Month-wise Sentiment Analysis')
        monthwise = df.groupby(['year', 'month', 'sentiment']).size().reset_index(name='count')
        palet = ['#FDD017', '#48a39e', '#045D5D']
        sns.set(style="whitegrid")
        unique_years = monthwise['year'].unique()
        nama_month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig, axes = plt.subplots(nrows=len(unique_years), ncols=1, figsize=(10, 4 * len(unique_years)))
        for i, year in enumerate(unique_years):
            monthwise_filtered = monthwise[monthwise['year'] == year]
            ax = sns.lineplot(x='month', y='count', hue='sentiment', data=monthwise_filtered, marker='o', palette=palet, ax=axes[i])
            for line, style in zip(ax.lines, ['--', '--', '--']):
                line.set_linestyle(style)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(nama_month)
            ax.set_title(f'Jumlah Ulasan Per Bulan Tahun {year}', fontsize=15)
            ax.set_xlabel('Bulan')
            ax.set_ylabel('Jumlah Ulasan')
            ax.grid(False)
            ax.legend(title='Sentimen', bbox_to_anchor=(1, 1), loc='upper left')
            for line in ax.lines:
                y_data = line.get_ydata()
                if len(y_data) > 0:
                    max_count_index = max(range(len(y_data)), key=y_data.__getitem__)
                    x_max, y_max = line.get_xdata()[max_count_index], y_data[max_count_index]
                    ax.annotate(f'Tertinggi: {y_max:.0f}', (x_max, y_max),
                                xytext=(10, -20),
                                textcoords='offset points',
                                arrowprops=dict(facecolor='g', arrowstyle='wedge,tail_width=0.7', alpha=0.5),
                                bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='#FDD017', alpha=0.5))
        plt.tight_layout()
        st.pyplot(fig)


        # Top Words Charts
        def plot_top_words(words, title):
            plt.figure(figsize=(12, 7))
            top_words = dict(Counter(words).most_common(25))
            bars = plt.barh(list(top_words.keys()), list(top_words.values()), color=plt.get_cmap('viridis')(np.linspace(0, 1, len(top_words))))
            plt.xlabel('Frequency')
            plt.title(title)
            plt.gca().invert_yaxis()  # To display the highest frequencies on top
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.0f}', va='center', ha='left', color='black', fontsize=10)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        st.subheader('Top Words - Sentiment Positive')
        top_words_positive = ' '.join([word for word in df[df['sentiment'] == 'Positive']['final_output_preprocessing']])
        plot_top_words(top_words_positive.split(), 'Top Words for Positive Sentiment')

        st.subheader('Top Words - Sentiment Negative')
        top_words_negative = ' '.join([word for word in df[df['sentiment'] == 'Negative']['final_output_preprocessing']])
        plot_top_words(top_words_negative.split(), 'Top Words for Negative Sentiment')

        st.subheader('Top Words - Sentiment Neutral')
        top_words_neutral = ' '.join([word for word in df[df['sentiment'] == 'Neutral']['final_output_preprocessing']])
        plot_top_words(top_words_neutral.split(), 'Top Words for Neutral Sentiment')
