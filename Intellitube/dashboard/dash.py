import os 
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import re
from collections import Counter      

# Set page config FIRST AND ONLY ONCE (at the very beginning)
st.set_page_config(
    page_title="YouTube Comment Analyzer 📊",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Then add parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure both ytcom and ytesenti  is correctly structured and accessible
try:
    from ytcom import extract_video_id
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


# Caching for performance
@st.cache_data(show_spinner="Fetching comments (this might take a moment)...")
def cached_fetch_comments(url, max_comments):
    return fetch_comments_scrape(url, max_comments=500) # Ensure max_comments is passed

@st.cache_data(show_spinner="Analyzing sentiment and intent...")
def cached_analyze_data(comments):
    sentiment_summary, sentiment_detailed = analyze_sentiment(comments)
    intent_summary, intent_detailed = analyze_intent(comments)
    return sentiment_summary, sentiment_detailed, intent_summary, intent_detailed


# Visualization functions
def plot_sentiment_distribution():
    if st.session_state.sentiment_data and st.session_state.sentiment_data['summary']:
        summary = st.session_state.sentiment_data['summary']
        df = pd.DataFrame.from_dict(summary, orient='index', columns=['Count'])
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Sentiment'}, inplace=True)
        
        # Ensure consistent ordering for colors
        sentiment_order = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
        df['Sentiment'] = pd.Categorical(df['Sentiment'], categories=sentiment_order, ordered=True)
        df = df.sort_values('Sentiment')

        fig = px.pie(df, values='Count', names='Sentiment', 
                     title='<b>Overall Sentiment Distribution</b>',
                     color='Sentiment',
                     color_discrete_map={
                         'POSITIVE': '#2ecc71', # Green
                         'NEUTRAL': '#f39c12',  # Orange
                         'NEGATIVE': '#e74c3c'  # Red
                     },
                     hole=0.4, # Deeper donut for better aesthetics
                     template="plotly_dark") # Use dark theme for Plotly
        
        fig.update_traces(textposition='inside', textinfo='percent+label', 
                          marker=dict(line=dict(color='#1a1a2e', width=2)), # Thicker border for slices
                          pull=[0.05 if s == df['Sentiment'].iloc[df['Count'].argmax()] else 0 for s in df['Sentiment']] # Pull out largest slice slightly
                          ) 
        fig.update_layout(
            showlegend=True, 
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
            font_color='#e0e0e0', # Font color for chart text
            legend=dict(font=dict(color='#e0e0e0', size=14)), # Legend font color and size
            title_font=dict(size=20), # Title font size
            margin=dict(l=20, r=20, t=60, b=20) # Adjust margins
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data to display. Please analyze comments first.")


def plot_intent_distribution():
    if st.session_state.intent_data and st.session_state.intent_data['summary']:
        summary = st.session_state.intent_data['summary']
        # Filter out 'N/A' if it exists and is not meaningful
        summary = {k: v for k, v in summary.items() if k != 'N/A' and v > 0}

        if not summary:
            st.info("No distinct intent data to display.")
            return

        df = pd.DataFrame.from_dict(summary, orient='index', columns=['Count'])
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Intent'}, inplace=True)
        df = df.sort_values('Count', ascending=False) # Sort by count for better readability
        
        fig = px.bar(df, x='Intent', y='Count', 
                     title='<b>Comment Intent Distribution</b>',
                     color='Intent',
                     color_discrete_sequence=px.colors.qualitative.D3, # Use a nice color palette
                     template="plotly_dark")
        fig.update_layout(
            xaxis_title="Intent Category", 
            yaxis_title="Number of Comments", 
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
            font_color='#e0e0e0', # Font color for chart text
            xaxis=dict(showgrid=False, tickfont=dict(color='#e0e0e0', size=12), title_font=dict(color='#e0e0e0', size=14)), # Hide grid and set tick/title color
            yaxis=dict(gridcolor='#3d3d5c', tickfont=dict(color='#e0e0e0', size=12), title_font=dict(color='#e0e0e0', size=14)), # Set grid color and tick/title color
            legend=dict(font=dict(color='#e0e0e0', size=12)), # Legend font color
            title_font=dict(size=20), # Title font size
            bargap=0.2 # Space between bars
        )
        fig.update_traces(marker_line_width=0) # No border for bars
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No intent data to display. Please analyze comments first.")

def plot_sentiment_score_distribution():
    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
        df = st.session_state.processed_df
        
        fig = px.histogram(df, x='Score', color='Sentiment',
                           title='<b>Distribution of Sentiment Scores</b>',
                           marginal='box', # Add a box plot for overall distribution
                           nbins=30, # More bins for detailed view
                           color_discrete_map={
                               'POSITIVE': '#2ecc71',
                               'NEUTRAL': '#f39c12',
                               'NEGATIVE': '#e74c3c'
                           },
                           template="plotly_dark",
                           hover_data={'Comment': True, 'Sentiment': True, 'Score': ':.3f'} # Show comment on hover
                           )
        fig.update_layout(
            bargap=0.1, 
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
            font_color='#e0e0e0', # Font color for chart text
            xaxis=dict(showgrid=False, tickfont=dict(color='#e0e0e0', size=12), title_font=dict(color='#e0e0e0', size=14), range=[-1, 1]), # Set fixed range for score
            yaxis=dict(gridcolor='#3d3d5c', tickfont=dict(color='#e0e0e0', size=12), title_font=dict(color='#e0e0e0', size=14)),
            legend=dict(font=dict(color='#e0e0e0', size=12)), # Legend font color
            title_font=dict(size=20), # Title font size
            xaxis_title="Sentiment Score", # Custom x-axis title
            yaxis_title="Number of Comments" # Custom y-axis title
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No processed data to display sentiment score distribution. Please analyze comments first.")

def generate_word_cloud(sentiment_type):
    if st.session_state.comments_data and st.session_state.sentiment_data and st.session_state.processed_df is not None:
        detailed_df = st.session_state.processed_df # Use the combined df
        
        # Filter comments by sentiment
        filtered_comments_df = detailed_df[detailed_df['Sentiment'] == sentiment_type]
        
        if filtered_comments_df.empty:
            st.warning(f"No {sentiment_type.lower()} comments available to generate a Word Cloud.")
            return
            
        text = ' '.join(filtered_comments_df['Comment'].astype(str)) # Ensure comments are strings
        
        # Add custom stop words (common YouTube phrases, etc.)
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(["video", "youtube", "comment", "like", "channel", "thanks", "great", "good", "https", "www", "com", "get", "dont", "just", "really", "much", "one", "can", "see", "also", "new", "time", "s", "t", "m", "u", "gonna", "want", "even", "would", "well", "go", "people", "make", "know", "said", "say", "every", "many", "way", "thi", "bro", "lmao", "lol"]) # Added more common web words and short words
        
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white', # Wordcloud image background must be distinct
                             colormap='viridis' if sentiment_type == 'POSITIVE' else 'Reds',
                             stopwords=custom_stopwords,
                             collocations=False, # Avoid combining common words like "New York"
                             max_words=100 # Limit to top 100 words
                             ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{sentiment_type} Comments Word Cloud', color='black', fontsize=16) # Title color for word cloud
        st.pyplot(fig, use_container_width=True)
    else:
        st.info(f"Please analyze comments to generate the {sentiment_type.lower()} Word Cloud.")


def plot_top_words(sentiment_type, n=15):
    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
        detailed_df = st.session_state.processed_df
        filtered_comments_df = detailed_df[detailed_df['Sentiment'] == sentiment_type]

        if filtered_comments_df.empty:
            st.info(f"No {sentiment_type.lower()} comments to analyze for top words.")
            return

        text = ' '.join(filtered_comments_df['Comment'].astype(str))
        words = re.findall(r'\b\w+\b', text.lower()) # Extract words
        
        # Define stopwords
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(["video", "youtube", "comment", "like", "channel", "thanks", "great", "good", "https", "www", "com", "get", "dont", "just", "really", "much", "one", "can", "see", "also", "new", "time", "s", "t", "m", "u", "gonna", "want", "even", "would", "well", "go", "people", "make", "know", "said", "say", "every", "many", "way", "thi", "bro", "lmao", "lol"])

        # Filter out stopwords and non-alphabetic words
        filtered_words = [word for word in words if word not in custom_stopwords and word.isalpha()]

        if not filtered_words:
            st.info(f"Not enough relevant words found in {sentiment_type.lower()} comments for top words plot.")
            return

        word_counts = Counter(filtered_words).most_common(n)
        words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])

        fig = px.bar(words_df.sort_values('Count', ascending=True), y='Word', x='Count',
                     title=f'<b>Top {n} Most Frequent Words in {sentiment_type} Comments</b>',
                     orientation='h',
                     color_discrete_sequence=['#8be9fd'], # A futuristic blue/cyan
                     template="plotly_dark")
        fig.update_layout(
            yaxis_title="", # Remove y-axis title as words are labels
            xaxis_title="Frequency", 
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            xaxis=dict(gridcolor='#3d3d5c', tickfont=dict(color='#e0e0e0', size=12), title_font=dict(color='#e0e0e0', size=14)),
            yaxis=dict(gridcolor='#3d3d5c', tickfont=dict(color='#e0e0e0', size=12), title_font=dict(color='#e0e0e0', size=14)),
            margin=dict(l=100), # Adjust left margin for long words
            title_font=dict(size=20) # Title font size
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Please analyze comments to plot Top Words for {sentiment_type.lower()} comments.")


def plot_comment_length_distribution():
    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
        df = st.session_state.processed_df
        df['Comment_Length'] = df['Comment'].apply(lambda x: len(str(x).split())) # Length in words

        fig = px.histogram(df, x='Comment_Length', 
                           title='<b>Distribution of Comment Lengths (Words)</b>',
                           nbins=50,
                           color_discrete_sequence=['#6272a4'], # Use a theme color
                           template="plotly_dark")
        fig.update_layout(
            xaxis_title="Number of Words", 
            yaxis_title="Number of Comments", 
            bargap=0.1, 
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            xaxis=dict(gridcolor='#3d3d5c', tickfont=dict(color='#e0e0e0', size=12), title_font=dict(color='#e0e0e0', size=14)),
            yaxis=dict(gridcolor='#3d3d5c', tickfont=dict(color='#e0e0e0', size=12), title_font=dict(color='#e0e0e0', size=14)),
            title_font=dict(size=20) # Title font size
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No processed data to display comment length distribution. Please analyze comments first.")


# Main analysis function
def analyze_video(video_id): # Function now takes video_id directly
    # Use a standard YouTube watch URL for fetching comments
    fetch_url = f"https://www.youtube.com/watch?v={video_id}" # This URL seems incorrect for actual YouTube API/scraping. It should ideally be a proper YouTube URL.
    # Corrected hypothetical fetch_url for illustration, assuming your backend `fetch_comments_scrape` can handle it:
    # fetch_url = f"https://www.youtube.com/watch?v={video_id}" 

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
        # Note: Merging by index is risky if detailed lists are not guaranteed to be in the same order.
        # A safer approach would be to include original comment text in both detailed outputs and merge on that.
        if len(intent_detailed) == len(df):
            # Assuming intent_detailed is a list of tuples like (comment_text, intent_label, score)
            # We need to extract the intent_label corresponding to each comment
            # A dictionary lookup is more robust if comments are unique:
            intent_map = {item[0]: item[1] for item in intent_detailed}
            df['Intent'] = df['Comment'].apply(lambda x: intent_map.get(x, 'N/A'))
        else:
            df['Intent'] = 'N/A' 
            st.warning("Could not perfectly merge intent data due to length mismatch or different comment ordering. Intent column might be incomplete.")
        
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

def show_comment_samples():
    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
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
            st.dataframe(filtered_df.sort_values(by='Score', ascending=(selected_sentiment == 'NEGATIVE')), # Sort negative comments by score ascending, positive/neutral descending
                         hide_index=True, use_container_width=True,
                         height=400) # Fixed height for better consistency
        else:
            st.info("No comments match the selected filters.")
    else:
        st.info("No processed comment data to display. Please analyze comments first.")

def show_top_comments(df, n=5):
    if df is not None and not df.empty:
        st.subheader(f"Top {n} Most Positive and Negative Comments")
        
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("##### Most Positive 😊")
            top_pos = df[df['Sentiment'] == 'POSITIVE'].sort_values('Score', ascending=False).head(n)
            if not top_pos.empty:
                for idx, row in top_pos.iterrows():
                    with st.container(border=True): # Use a container for each comment
                        st.markdown(f"**Score:** `{row['Score']:.3f}`")
                        st.markdown(f"*{row['Comment']}*")
            else:
                st.info("No positive comments found.")
        
        with col_neg:
            st.markdown("##### Most Negative 😠")
            top_neg = df[df['Sentiment'] == 'NEGATIVE'].sort_values('Score', ascending=True).head(n)
            if not top_neg.empty:
                for idx, row in top_neg.iterrows():
                    with st.container(border=True): # Use a container for each comment
                        st.markdown(f"**Score:** `{row['Score']:.3f}`")
                        st.markdown(f"*{row['Comment']}*")
            else:
                st.info("No negative comments found.")
    else:
        st.info("No processed data to display top comments. Please analyze comments first.")


# Main app function
def main():
    st.title("📊 Comment Analytics ")
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
                st.write(f"Fetching and processing data for video ID: **`{video_id}`**")
                if analyze_video(video_id): # Pass video_id directly to analyze_video
                    status.update(label="Analysis complete! ✨", state="complete", expanded=False)
                    st.markdown("---") # Visual separator after analysis complete
                    display_results()
                else:
                    status.update(label="Analysis failed! 🚨", state="error", expanded=True)
                    st.error("Failed to perform analysis. Please ensure the video ID is valid and has comments enabled.") # More specific error
    else:
        st.warning("No video ID provided in the URL. Please launch this dashboard from the Gradio app.")
        st.info("The Gradio app will pass the video ID for analysis automatically via URL parameters. Please ensure you have started the Gradio app first.")

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
            st.metric(label="Positive Comments", value=f"{summary.get('POSITIVE', 0)}", 
                      delta=f"{summary.get('POSITIVE', 0)/total:.1%}" if total > 0 else "0.0%")
        
        with col2:
            st.metric(label="Neutral Comments", value=f"{summary.get('NEUTRAL', 0)}", 
                      delta=f"{summary.get('NEUTRAL', 0)/total:.1%}" if total > 0 else "0.0%")
        
        with col3:
            st.metric(label="Negative Comments", value=f"{summary.get('NEGATIVE', 0)}", 
                      delta=f"{summary.get('NEGATIVE', 0)/total:.1%}" if total > 0 else "0.0%")

    st.markdown("---") # Horizontal line for separation

    # Charts section - Using Tabs for structure
    st.subheader("Visualizations")
    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(["Sentiment & Intent", "Sentiment Scores", "Text Insights", "Comment Samples"])
    
    with chart_tab1:
        st.markdown("### Sentiment and Intent Distributions")
        col_sent, col_intent = st.columns(2)
        with col_sent:
            with st.container(border=True):
                plot_sentiment_distribution()
        with col_intent:
            with st.container(border=True):
                plot_intent_distribution()
    
    with chart_tab2:
        st.markdown("### Detailed Sentiment Score Analysis")
        with st.container(border=True):
            plot_sentiment_score_distribution()

    with chart_tab3:
        st.markdown("### Textual Insights from Comments")
        st.markdown("#### Word Clouds")
        wordcloud_col1, wordcloud_col2 = st.columns(2)
        with wordcloud_col1:
            with st.container(border=True):
                generate_word_cloud('POSITIVE')
        with wordcloud_col2:
            with st.container(border=True):
                generate_word_cloud('NEGATIVE')
        
        st.markdown("---") # Separator for word clouds and top words
        st.markdown("#### Most Frequent Words")
        top_words_col1, top_words_col2 = st.columns(2)
        with top_words_col1:
            with st.container(border=True):
                plot_top_words('POSITIVE')
        with top_words_col2:
            with st.container(border=True):
                plot_top_words('NEGATIVE')

        st.markdown("---") # Separator for top words and comment length
        st.markdown("#### Comment Length Analysis")
        with st.container(border=True):
            plot_comment_length_distribution()


    with chart_tab4:
        # Top comments section
        with st.container(border=True):
            if st.session_state.processed_df is not None:
                show_top_comments(st.session_state.processed_df)
            else:
                st.info("No processed data to display top comments. Please analyze comments first.")


        st.markdown("---") # Horizontal line for separation

        # Comment samples (now includes filtering)
        with st.container(border=True):
            show_comment_samples()
    
    st.markdown("---") # Horizontal line for separation
    st.info("Analysis powered by advanced NLP models. If comments cannot be fetched, ensure the video is public and has comments enabled.")

if __name__ == "__main__":
    main()