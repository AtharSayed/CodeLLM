import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path
import re

# Set page config FIRST AND ONLY ONCE (at the very beginning)
st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Then add parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ytsenti import fetch_comments_scrape, analyze_sentiment, analyze_intent

# Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Session state initialization
if 'comments_data' not in st.session_state:
    st.session_state.comments_data = None
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = None
if 'intent_data' not in st.session_state:
    st.session_state.intent_data = None

# Helper functions
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def is_valid_youtube_url(url):
    if not url:
        return False
    youtube_regex = re.compile(
        r"(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+"
    )
    return youtube_regex.match(url) is not None

# Main analysis function
def analyze_video(url):
    try:
        with st.spinner('Fetching comments...'):
            comments = fetch_comments_scrape(url, max_comments=100)
        
        if not comments:
            st.error("No comments found or couldn't fetch comments.")
            return False
            
        with st.spinner('Analyzing sentiment...'):
            sentiment_summary, sentiment_detailed = analyze_sentiment(comments)
            intent_summary, intent_detailed = analyze_intent(comments)
            
        # Store in session state
        st.session_state.comments_data = comments
        st.session_state.sentiment_data = {
            'summary': sentiment_summary,
            'detailed': sentiment_detailed
        }
        st.session_state.intent_data = {
            'summary': intent_summary,
            'detailed': intent_detailed
        }
        
        return True
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return False

# Visualization functions
def plot_sentiment_distribution():
    if st.session_state.sentiment_data:
        summary = st.session_state.sentiment_data['summary']
        df = pd.DataFrame.from_dict(summary, orient='index', columns=['Count'])
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Sentiment'}, inplace=True)
        
        fig = px.pie(df, values='Count', names='Sentiment', 
                     title='Sentiment Distribution',
                     color='Sentiment',
                     color_discrete_map={
                         'POSITIVE': '#2ecc71',
                         'NEUTRAL': '#f39c12',
                         'NEGATIVE': '#e74c3c'
                     })
        st.plotly_chart(fig, use_container_width=True)

def plot_intent_distribution():
    if st.session_state.intent_data:
        summary = st.session_state.intent_data['summary']
        df = pd.DataFrame.from_dict(summary, orient='index', columns=['Count'])
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Intent'}, inplace=True)
        
        fig = px.bar(df, x='Intent', y='Count', 
                     title='Comment Intent Distribution',
                     color='Intent')
        st.plotly_chart(fig, use_container_width=True)

def generate_word_cloud(sentiment_type):
    if st.session_state.comments_data and st.session_state.sentiment_data:
        comments = st.session_state.comments_data
        detailed = st.session_state.sentiment_data['detailed']
        
        # Filter comments by sentiment
        filtered_comments = [
            comment for comment, (_, label, _) in zip(comments, detailed) 
            if label == sentiment_type
        ]
        
        if not filtered_comments:
            st.warning(f"No {sentiment_type.lower()} comments to display")
            return
            
        text = ' '.join(filtered_comments)
        
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis' if sentiment_type == 'POSITIVE' else 'Reds').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{sentiment_type} Comments Word Cloud')
        st.pyplot(fig)

def show_comment_samples():
    if st.session_state.comments_data and st.session_state.sentiment_data:
        detailed = st.session_state.sentiment_data['detailed']
        
        # Create DataFrame
        df = pd.DataFrame(detailed, columns=['Comment', 'Sentiment', 'Score'])
        df['Score'] = df['Score'].round(3)
        
        # Display samples
        st.subheader("Comment Samples")
        
        tabs = st.tabs(["All", "Positive", "Neutral", "Negative"])
        
        with tabs[0]:
            st.dataframe(df.sort_values('Score', ascending=False), 
                        hide_index=True, use_container_width=True)
        
        with tabs[1]:
            pos_df = df[df['Sentiment'] == 'POSITIVE']
            st.dataframe(pos_df.sort_values('Score', ascending=False), 
                         hide_index=True, use_container_width=True)
        
        with tabs[2]:
            neu_df = df[df['Sentiment'] == 'NEUTRAL']
            st.dataframe(neu_df.sort_values('Score', ascending=False), 
                         hide_index=True, use_container_width=True)
        
        with tabs[3]:
            neg_df = df[df['Sentiment'] == 'NEGATIVE']
            st.dataframe(neg_df.sort_values('Score', ascending=False), 
                         hide_index=True, use_container_width=True)

# Main app function (REMOVED duplicate set_page_config from here)
def main():
    st.title("📊 YouTube Sentiment Analysis Dashboard")
    
    # Get video_id from URL parameters
    params = st.experimental_get_query_params()
    video_id = params.get("video_id", [None])[0]
    
    if video_id:
        url = f"https://www.youtube.com/watch?v={video_id}"
        st.success(f"Analyzing: {url}")
        
        # Automatically run analysis if not already done
        if st.session_state.comments_data is None:
            if analyze_video(url):
                display_results()
        else:
            display_results()
    else:
        st.warning("No video specified. Please launch from Gradio app.")

def display_results():
    """Display all analysis results"""
    # Overall metrics
    st.subheader("Overall Sentiment")
    col1, col2, col3 = st.columns(3)
    
    summary = st.session_state.sentiment_data['summary']
    total = sum(summary.values())
    
    with col1:
        st.metric("Positive", f"{summary['POSITIVE']} ({summary['POSITIVE']/total:.1%})")
    
    with col2:
        st.metric("Neutral", f"{summary['NEUTRAL']} ({summary['NEUTRAL']/total:.1%})")
    
    with col3:
        st.metric("Negative", f"{summary['NEGATIVE']} ({summary['NEGATIVE']/total:.1%})")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        plot_sentiment_distribution()
    
    with col2:
        plot_intent_distribution()
    
    # Word clouds
    st.subheader("Comment Word Clouds")
    tabs = st.tabs(["Positive", "Negative"])
    
    with tabs[0]:
        generate_word_cloud('POSITIVE')
    
    with tabs[1]:
        generate_word_cloud('NEGATIVE')
    
    # Comment samples
    show_comment_samples()

if __name__ == "__main__":
    main()