import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import re

# Set page config FIRST AND ONLY ONCE (at the very beginning)
st.set_page_config(
    page_title="YouTube Comment Analyzer 📊",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Then add parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Ensure 'ytsenti' is correctly structured and accessible
try:
    from ytsenti import fetch_comments_scrape, analyze_sentiment, analyze_intent
except ImportError:
    st.error("Could not import 'ytsenti'. Make sure 'ytsenti.py' is in the parent directory.")
    st.stop()

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
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None # Store processed DataFrame
if 'current_video_id' not in st.session_state: # Track the video ID being displayed
    st.session_state.current_video_id = None


# Helper functions
def extract_video_id(url):
    # Updated regex for more robust ID extraction
    match = re.search(r"(?:v=|youtu\.be/|embed/|live/|watch\?v=)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

# Caching for performance
@st.cache_data(show_spinner="Fetching comments (this might take a moment)...")
def cached_fetch_comments(url, max_comments):
    return fetch_comments_scrape(url, max_comments=100) # Ensure max_comments is passed

@st.cache_data(show_spinner="Analyzing sentiment and intent...")
def cached_analyze_data(comments):
    sentiment_summary, sentiment_detailed = analyze_sentiment(comments)
    intent_summary, intent_detailed = analyze_intent(comments)
    return sentiment_summary, sentiment_detailed, intent_summary, intent_detailed


# Main analysis function
def analyze_video(video_id): # Function now takes video_id directly
    # Use a standard YouTube watch URL for fetching comments
    fetch_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        comments = cached_fetch_comments(fetch_url, max_comments=100) # Using cached function
        
        if not comments:
            st.error("No comments found or couldn't fetch comments. Please check the video ID or try again.")
            return False
            
        sentiment_summary, sentiment_detailed, intent_summary, intent_detailed = cached_analyze_data(comments) # Using cached function
            
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

        # Create a combined DataFrame for easier filtering and display
        df = pd.DataFrame(sentiment_detailed, columns=['Comment', 'Sentiment', 'Score'])
        df['Score'] = df['Score'].round(3)
        
        # Merge intent data if available and aligned (assuming same order for simplicity)
        if len(intent_detailed) == len(df):
            df['Intent'] = [item[1] for item in intent_detailed] # Assuming intent_detailed is list of (comment, intent_label, score)
        else:
            st.warning("Could not perfectly merge intent data due to length mismatch.")
            df['Intent'] = 'N/A' # Fallback
        
        st.session_state.processed_df = df
        st.session_state.current_video_id = video_id # Store the ID that was successfully analyzed
        
        return True
        
    except ValueError as ve:
        st.error(f"Invalid Video ID: {str(ve)}. Please ensure the video ID is correct.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis: {str(e)}")
        st.info("Please ensure the video exists, is publicly accessible, and has comments enabled.")
        return False

# Visualization functions
def plot_sentiment_distribution():
    if st.session_state.sentiment_data:
        summary = st.session_state.sentiment_data['summary']
        df = pd.DataFrame.from_dict(summary, orient='index', columns=['Count'])
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Sentiment'}, inplace=True)
        
        fig = px.pie(df, values='Count', names='Sentiment', 
                     title='<b>Overall Sentiment Distribution</b>',
                     color='Sentiment',
                     color_discrete_map={
                         'POSITIVE': '#2ecc71', # Green
                         'NEUTRAL': '#f39c12',  # Orange
                         'NEGATIVE': '#e74c3c'  # Red
                     },
                     hole=0.3) # Donut chart for better aesthetics
        fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#1a1a2e', width=1))) # Add border to pie slices
        fig.update_layout(
            showlegend=True, 
            title_x=0.5,
            plot_bgcolor='#2a2a47', # Chart background
            paper_bgcolor='#2a2a47', # Paper background
            font_color='#e0e0e0', # Font color for chart text
            legend=dict(font=dict(color='#e0e0e0')) # Legend font color
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_intent_distribution():
    if st.session_state.intent_data:
        summary = st.session_state.intent_data['summary']
        df = pd.DataFrame.from_dict(summary, orient='index', columns=['Count'])
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Intent'}, inplace=True)
        
        fig = px.bar(df, x='Intent', y='Count', 
                     title='<b>Comment Intent Distribution</b>',
                     color='Intent',
                     color_discrete_sequence=px.colors.qualitative.D3) # Use a nice color palette
        fig.update_layout(
            xaxis_title="Intent Category", 
            yaxis_title="Number of Comments", 
            title_x=0.5,
            plot_bgcolor='#2a2a47', # Chart background
            paper_bgcolor='#2a2a47', # Paper background
            font_color='#e0e0e0', # Font color for chart text
            xaxis=dict(showgrid=False, tickfont=dict(color='#e0e0e0'), title_font=dict(color='#e0e0e0')), # Hide grid and set tick/title color
            yaxis=dict(gridcolor='#3d3d5c', tickfont=dict(color='#e0e0e0'), title_font=dict(color='#e0e0e0')), # Set grid color and tick/title color
            legend=dict(font=dict(color='#e0e0e0')) # Legend font color
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_score_distribution():
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        
        fig = px.histogram(df, x='Score', color='Sentiment',
                           title='<b>Distribution of Sentiment Scores</b>',
                           marginal='box', # Add a box plot for overall distribution
                           nbins=30, # More bins for detailed view
                           color_discrete_map={
                               'POSITIVE': '#2ecc71',
                               'NEUTRAL': '#f39c12',
                               'NEGATIVE': '#e74c3c'
                           })
        fig.update_layout(
            bargap=0.1, 
            title_x=0.5,
            plot_bgcolor='#2a2a47', # Chart background
            paper_bgcolor='#2a2a47', # Paper background
            font_color='#e0e0e0', # Font color for chart text
            xaxis=dict(showgrid=False, tickfont=dict(color='#e0e0e0'), title_font=dict(color='#e0e0e0')),
            yaxis=dict(gridcolor='#3d3d5c', tickfont=dict(color='#e0e0e0'), title_font=dict(color='#e0e0e0')),
            legend=dict(font=dict(color='#e0e0e0')) # Legend font color
        )
        st.plotly_chart(fig, use_container_width=True)

def generate_word_cloud(sentiment_type):
    if st.session_state.comments_data and st.session_state.sentiment_data:
        detailed_df = st.session_state.processed_df # Use the combined df
        
        # Filter comments by sentiment
        filtered_comments_df = detailed_df[detailed_df['Sentiment'] == sentiment_type]
        
        if filtered_comments_df.empty:
            st.warning(f"No {sentiment_type.lower()} comments to display for Word Cloud.")
            return
            
        text = ' '.join(filtered_comments_df['Comment'])
        
        # Add custom stop words (common YouTube phrases, etc.)
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(["video", "youtube", "comment", "like", "channel", "thanks", "great", "good", "https", "www", "com", "get", "dont", "just", "really", "much", "one", "can", "see", "also"]) # Added more common web words and short words
        
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white', # Wordcloud image background must be distinct
                             colormap='viridis' if sentiment_type == 'POSITIVE' else 'Reds',
                             stopwords=custom_stopwords,
                             collocations=False # Avoid combining common words like "New York"
                             ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{sentiment_type} Comments Word Cloud', color='black') # Title color for word cloud
        st.pyplot(fig)

def show_comment_samples():
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        
        st.subheader("Comment Samples and Details")
        
        # Add filters for viewing comments
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            selected_sentiment = st.selectbox(
                "Filter by Sentiment:", 
                ["All", "POSITIVE", "NEUTRAL", "NEGATIVE"],
                key="comment_sentiment_filter"
            )
        with col_filter2:
            # Check if 'Intent' column exists before offering filter
            if 'Intent' in df.columns and df['Intent'].nunique() > 1:
                unique_intents = ['All'] + sorted(df['Intent'].unique().tolist())
                selected_intent = st.selectbox(
                    "Filter by Intent:", 
                    unique_intents,
                    key="comment_intent_filter"
                )
            else:
                selected_intent = "All" # No intent filter if column doesn't exist or only one unique value


        filtered_df = df.copy()

        if selected_sentiment != "All":
            filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]
        if selected_intent != "All" and 'Intent' in df.columns:
            filtered_df = filtered_df[filtered_df['Intent'] == selected_intent]

        # Display filtered data with sorting options
        if not filtered_df.empty:
            st.dataframe(filtered_df.sort_values(by='Score', ascending=(selected_sentiment != 'POSITIVE')), 
                         hide_index=True, use_container_width=True,
                         height=400) # Fixed height for better consistency
        else:
            st.info("No comments match the selected filters.")

def show_top_comments(df, n=5):
    if df is not None and not df.empty:
        st.subheader(f"Top {n} Most Positive and Negative Comments")
        
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("##### Most Positive 😊")
            top_pos = df[df['Sentiment'] == 'POSITIVE'].sort_values('Score', ascending=False).head(n)
            if not top_pos.empty:
                for idx, row in top_pos.iterrows():
                    st.success(f"**Score:** {row['Score']:.3f}\n\n{row['Comment']}\n---")
            else:
                st.info("No positive comments found.")
        
        with col_neg:
            st.markdown("##### Most Negative 😠")
            top_neg = df[df['Sentiment'] == 'NEGATIVE'].sort_values('Score', ascending=True).head(n)
            if not top_neg.empty:
                for idx, row in top_neg.iterrows():
                    st.error(f"**Score:** {row['Score']:.3f}\n\n{row['Comment']}\n---")
            else:
                st.info("No negative comments found.")

# Main app function
def main():
    st.title("📊 YouTube Comment Analyzer")
    st.markdown("---")
    
    # Get video_id from URL parameters using st.query_params
    params = st.query_params
    video_id = params.get("video_id", None)
    
    if video_id:
        # Check if we're already displaying results for this video_id
        if st.session_state.current_video_id == video_id and st.session_state.sentiment_data is not None:
            # Already analyzed and displayed, just show the results
            display_results()
        else:
            # New video_id or not yet analyzed
            # Clear previous session data if it's a new video_id
            if st.session_state.current_video_id != video_id:
                st.session_state.comments_data = None
                st.session_state.sentiment_data = None
                st.session_state.intent_data = None
                st.session_state.processed_df = None
                st.session_state.current_video_id = None # Reset until successful analysis
            
            # Use st.status for a cleaner, temporary message during analysis
            with st.status("Analyzing comments...", expanded=True) as status:
                st.write(f"Fetching and processing data for video ID: **{video_id}**")
                if analyze_video(video_id): # Pass video_id directly to analyze_video
                    status.update(label="Analysis complete!", state="complete", expanded=False)
                    st.markdown("---") # Visual separator after analysis complete
                    display_results()
                else:
                    status.update(label="Analysis failed!", state="error", expanded=True)
                    st.error("Failed to perform analysis. Please ensure the video ID is valid and has comments enabled.") # More specific error
    else:
        st.warning("No video ID provided in the URL. Please launch this dashboard from the Gradio app.")
        st.info("The Gradio app will pass the video ID for analysis automatically via URL parameters.")

def display_results():
    """Display all analysis results in a structured and professional layout."""
    
    st.header("Analysis Results ✨")

    # Overall metrics in a streamlined container
    with st.container(border=True):
        st.subheader("Overall Sentiment Summary")
        col1, col2, col3 = st.columns(3)
        
        summary = st.session_state.sentiment_data['summary']
        total = sum(summary.values())
        
        # Use a consistent display for metrics
        with col1:
            st.metric(label="Positive Comments", value=f"{summary.get('POSITIVE', 0)}", delta=f"{summary.get('POSITIVE', 0)/total:.1%}")
        
        with col2:
            st.metric(label="Neutral Comments", value=f"{summary.get('NEUTRAL', 0)}", delta=f"{summary.get('NEUTRAL', 0)/total:.1%}")
        
        with col3:
            st.metric(label="Negative Comments", value=f"{summary.get('NEGATIVE', 0)}", delta=f"{summary.get('NEGATIVE', 0)/total:.1%}")

    st.markdown("---") # Horizontal line for separation

    # Charts section
    st.subheader("Visualizations")
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Sentiment Distribution", "Intent Distribution", "Score Distribution"])
    
    with chart_tab1:
        plot_sentiment_distribution()
    
    with chart_tab2:
        plot_intent_distribution()

    with chart_tab3:
        plot_sentiment_score_distribution()
    
    st.markdown("---") # Horizontal line for separation

    # Word clouds
    st.subheader("Insightful Word Clouds")
    wordcloud_col1, wordcloud_col2 = st.columns(2)
    
    with wordcloud_col1:
        with st.container(border=True): # Use container for word clouds
            generate_word_cloud('POSITIVE')
    
    with wordcloud_col2:
        with st.container(border=True): # Use container for word clouds
            generate_word_cloud('NEGATIVE')
    
    st.markdown("---") # Horizontal line for separation

    # Top comments section
    if st.session_state.processed_df is not None:
        show_top_comments(st.session_state.processed_df)

    st.markdown("---") # Horizontal line for separation

    # Comment samples (now includes filtering)
    show_comment_samples()
    
    st.markdown("---") # Horizontal line for separation
    st.info("Analysis powered by advanced NLP models.")

if __name__ == "__main__":
    main()