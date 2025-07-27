import os
import sys
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from urllib.parse import unquote
import time

# Page configuration
st.set_page_config(
    page_title="YouTube Comment Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced styling
st.markdown("""
    <style>
    /* Dark theme with modern gradients */
    .main { background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%); }
    
    /* Beautiful metrics */
    .metric-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #3d5afe;
        box-shadow: 0 8px 32px rgba(61, 90, 254, 0.2);
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
    }
    
    /* Enhanced containers */
    .stContainer > div {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 15px;
        border: 1px solid #44475a;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Modern tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #232526 0%, #414345 100%);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        color: #f8f8f2;
        font-weight: 500;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #8be9fd;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced selectboxes */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        border-radius: 10px;
        border: 1px solid #6272a4;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        background-size: 400% 400%;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Headers */
    h1, h2, h3 { 
        color: #f8f8f2; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Custom cards */
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #8be9fd;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Expandable sections */
    .stExpander {
        background: rgba(30, 30, 46, 0.6);
        border: 1px solid #44475a;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Try to import required modules with better error handling
try:
    from ytcom import extract_video_id, fetch_comments_scrape
    from ytsenti import analyze_sentiment, analyze_intent
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Make sure all required files (ytcom.py, ytsenti.py) are in the correct directory.")
    st.stop()

# Enhanced session state initialization
for key in ['comments_data', 'sentiment_data', 'intent_data', 'processed_df', 
           'current_video_id', 'analysis_metadata', 'filtered_df']:
    if key not in st.session_state:
        st.session_state[key] = None

# Enhanced caching with better error handling
@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch_comments(url, max_comments):
    """Cached function to fetch comments with error handling"""
    try:
        comments = fetch_comments_scrape(url, max_comments=max_comments)
        if not comments:
            return []
        if isinstance(comments, list) and len(comments) == 1 and comments[0].startswith("❌"):
            return []
        return comments
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def cached_analyze_data(comments):
    """Cached analysis with comprehensive error handling"""
    try:
        # Validate input
        if not comments or not isinstance(comments, list):
            st.error("Invalid comments data provided to analysis function")
            return {}, [], {}, []
        
        if len(comments) == 0:
            st.error("No comments to analyze")
            return {}, [], {}, []
        
        # Analyze sentiment
        try:
            sentiment_result = analyze_sentiment(comments)
            if sentiment_result is None:
                st.error("Sentiment analysis returned None")
                return {}, [], {}, []
            
            if isinstance(sentiment_result, tuple) and len(sentiment_result) == 2:
                sentiment_summary, sentiment_detailed = sentiment_result
            else:
                st.error(f"Unexpected sentiment analysis format: {type(sentiment_result)}")
                return {}, [], {}, []
                
        except Exception as e:
            st.error(f"Error in sentiment analysis: {e}")
            return {}, [], {}, []
        
        # Analyze intent
        try:
            intent_result = analyze_intent(comments)
            if intent_result is None:
                st.error("Intent analysis returned None")
                return sentiment_summary or {}, sentiment_detailed or [], {}, []
            
            if isinstance(intent_result, tuple) and len(intent_result) == 2:
                intent_summary, intent_detailed = intent_result
            else:
                st.error(f"Unexpected intent analysis format: {type(intent_result)}")
                return sentiment_summary or {}, sentiment_detailed or [], {}, []
                
        except Exception as e:
            st.error(f"Error in intent analysis: {e}")
            return sentiment_summary or {}, sentiment_detailed or [], {}, []
        
        # Validate results
        if sentiment_summary is None:
            sentiment_summary = {}
        if sentiment_detailed is None:
            sentiment_detailed = []
        if intent_summary is None:
            intent_summary = {}
        if intent_detailed is None:
            intent_detailed = []
            
        return sentiment_summary, sentiment_detailed, intent_summary, intent_detailed
        
    except Exception as e:
        st.error(f"Critical error in analysis: {e}")
        return {}, [], {}, []

def create_enhanced_metrics():
    """Create beautiful metric cards with additional insights"""
    if not st.session_state.sentiment_data or not st.session_state.sentiment_data.get('summary'):
        st.warning("No sentiment data available for metrics")
        return
    
    summary = st.session_state.sentiment_data['summary']
    total = sum(summary.values()) if summary else 0
    
    if total == 0:
        st.warning("No sentiment data to display")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        positive_pct = (summary.get('POSITIVE', 0) / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; color:#2ecc71;">😊 Positive</h3>
            <h2 style="margin:0.5rem 0; color:white;">{summary.get('POSITIVE', 0)}</h2>
            <p style="margin:0; color:#a8dadc;">{positive_pct:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        neutral_pct = (summary.get('NEUTRAL', 0) / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; color:#f39c12;">😐 Neutral</h3>
            <h2 style="margin:0.5rem 0; color:white;">{summary.get('NEUTRAL', 0)}</h2>
            <p style="margin:0; color:#a8dadc;">{neutral_pct:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        negative_pct = (summary.get('NEGATIVE', 0) / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; color:#e74c3c;">😞 Negative</h3>
            <h2 style="margin:0.5rem 0; color:white;">{summary.get('NEGATIVE', 0)}</h2>
            <p style="margin:0; color:#a8dadc;">{negative_pct:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_score = 0
        if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
            avg_score = st.session_state.processed_df['Score'].mean()
        
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; color:#8be9fd;">📊 Total</h3>
            <h2 style="margin:0.5rem 0; color:white;">{total}</h2>
            <p style="margin:0; color:#a8dadc;">Avg Score: {avg_score:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

def plot_enhanced_sentiment_distribution():
    """Enhanced sentiment distribution with animations and interactivity"""
    if not st.session_state.sentiment_data or not st.session_state.sentiment_data.get('summary'):
        st.info("No sentiment data available for distribution plot")
        return
    
    summary = st.session_state.sentiment_data['summary']
    
    if not summary or sum(summary.values()) == 0:
        st.info("No sentiment data to visualize")
        return
    
    total = sum(summary.values())
    
    # Create subplot with donut chart and bar chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sentiment Distribution', 'Sentiment Breakdown'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Enhanced donut chart
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = list(summary.keys())
    values = list(summary.values())
    
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=colors[:len(labels)], line=dict(color='#1a1a2e', width=3)),
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.1 if v == max(values) else 0 for v in values]
        ),
        row=1, col=1
    )
    
    # Add center text to donut
    fig.add_annotation(
        text=f"<b>{total}</b><br>Total<br>Comments",
        x=0.18, y=0.5, font_size=16, showarrow=False,
        font=dict(color='white')
    )
    
    # Enhanced bar chart
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors[:len(labels)],
                line=dict(color='#1a1a2e', width=2)
            ),
            text=[f'{v}<br>({v/total*100:.1f}%)' for v in values],
            textposition='auto',
            textfont=dict(color='white', size=12),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title=dict(text="Sentiment Analysis Overview", x=0.5, font=dict(size=20))
    )
    
    fig.update_xaxes(title_text="Sentiment Category", row=1, col=2)
    fig.update_yaxes(title_text="Number of Comments", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_enhanced_intent_analysis():
    """Enhanced intent analysis with hierarchical visualization"""
    if not st.session_state.intent_data or not st.session_state.intent_data.get('summary'):
        st.info("No intent data available")
        return
    
    summary = st.session_state.intent_data['summary']
    # Filter meaningful intents
    summary = {k: v for k, v in summary.items() if k != 'N/A' and v > 0}
    
    if not summary:
        st.info("No meaningful intent data to display")
        return
    
    # Sort by count
    sorted_intents = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=list(sorted_intents.values()),
        y=list(sorted_intents.keys()),
        orientation='h',
        marker=dict(
            color=list(sorted_intents.values()),
            colorscale='Viridis',
            line=dict(color='#1a1a2e', width=2)
        ),
        text=[f'{v} ({v/sum(sorted_intents.values())*100:.1f}%)' for v in sorted_intents.values()],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Percentage: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title=dict(text="Intent Classification Analysis", x=0.5, font=dict(size=20)),
        xaxis_title="Number of Comments",
        yaxis_title="Intent Type",
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_trend_analysis():
    """Advanced trend analysis with moving averages and confidence bands"""
    if st.session_state.processed_df is None or st.session_state.processed_df.empty:
        st.info("No data available for trend analysis")
        return
    
    df = st.session_state.processed_df.copy()
    
    # Create batches for trend analysis
    batch_size = st.slider("Batch Size for Trend Analysis", 10, 100, 25, step=5)
    df['Batch'] = (df.index // batch_size) + 1
    
    # Calculate sentiment percentages and moving averages
    trend_data = df.groupby(['Batch', 'Sentiment']).size().unstack(fill_value=0)
    trend_pct = trend_data.div(trend_data.sum(axis=1), axis=0) * 100
    
    # Calculate moving averages
    window = max(3, len(trend_pct) // 5)
    trend_ma = trend_pct.rolling(window=window, center=True).mean()
    
    fig = go.Figure()
    
    colors = {'POSITIVE': '#2ecc71', 'NEUTRAL': '#f39c12', 'NEGATIVE': '#e74c3c'}
    
    for sentiment in trend_pct.columns:
        # Add main trend line
        fig.add_trace(go.Scatter(
            x=trend_pct.index,
            y=trend_pct[sentiment],
            mode='lines+markers',
            name=f'{sentiment} (Raw)',
            line=dict(color=colors.get(sentiment, '#8be9fd'), width=2),
            opacity=0.6
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=trend_ma.index,
            y=trend_ma[sentiment],
            mode='lines',
            name=f'{sentiment} (Trend)',
            line=dict(color=colors.get(sentiment, '#8be9fd'), width=4),
        ))
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(text="Sentiment Trends Across Comment Batches", x=0.5, font=dict(size=20)),
        xaxis_title=f"Comment Batch ({batch_size} comments each)",
        yaxis_title="Percentage (%)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_sentiment_score_analysis():
    """Enhanced sentiment score analysis with statistical insights"""
    if st.session_state.processed_df is None or st.session_state.processed_df.empty:
        st.info("No data available for score analysis")
        return
    
    df = st.session_state.processed_df
    
    # Create subplot with multiple visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Score Distribution', 'Box Plot by Sentiment', 
                       'Score vs Comment Length', 'Confidence Heatmap'),
        specs=[[{"type": "histogram"}, {"type": "box"}],
               [{"type": "scatter"}, {"type": "heatmap"}]]
    )
    
    colors = {'POSITIVE': '#2ecc71', 'NEUTRAL': '#f39c12', 'NEGATIVE': '#e74c3c'}
    
    # 1. Score distribution histogram
    for sentiment in df['Sentiment'].unique():
        sentiment_data = df[df['Sentiment'] == sentiment]
        fig.add_trace(
            go.Histogram(
                x=sentiment_data['Score'],
                name=sentiment,
                opacity=0.7,
                marker_color=colors.get(sentiment, '#8be9fd'),
                nbinsx=30
            ),
            row=1, col=1
        )
    
    # 2. Box plot by sentiment
    for sentiment in df['Sentiment'].unique():
        sentiment_data = df[df['Sentiment'] == sentiment]
        fig.add_trace(
            go.Box(
                y=sentiment_data['Score'],
                name=sentiment,
                marker_color=colors.get(sentiment, '#8be9fd'),
                boxpoints='outliers'
            ),
            row=1, col=2
        )
    
    # 3. Score vs Comment Length scatter
    df['Comment_Length'] = df['Comment'].str.len()
    for sentiment in df['Sentiment'].unique():
        sentiment_data = df[df['Sentiment'] == sentiment]
        fig.add_trace(
            go.Scatter(
                x=sentiment_data['Comment_Length'],
                y=sentiment_data['Score'],
                mode='markers',
                name=sentiment,
                marker=dict(
                    color=colors.get(sentiment, '#8be9fd'),
                    size=8,
                    opacity=0.7
                )
            ),
            row=2, col=1
        )
    
    # 4. Confidence heatmap
    score_bins = pd.cut(df['Score'], bins=10, labels=[f'{i/10:.1f}' for i in range(10)])
    heatmap_data = pd.crosstab(df['Sentiment'], score_bins, normalize='index') * 100
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            showscale=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=800,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(text="Comprehensive Sentiment Score Analysis", x=0.5, font=dict(size=20))
    )
    
    st.plotly_chart(fig, use_container_width=True)

def generate_enhanced_wordcloud(sentiment_type):
    """Enhanced word cloud with better preprocessing and styling"""
    if st.session_state.processed_df is None or st.session_state.processed_df.empty:
        st.info("No data available for word cloud")
        return
    
    df = st.session_state.processed_df
    sentiment_comments = df[df['Sentiment'] == sentiment_type]['Comment']
    
    if sentiment_comments.empty:
        st.info(f"No {sentiment_type.lower()} comments available")
        return
    
    # Enhanced text preprocessing
    text = ' '.join(sentiment_comments.astype(str))
    
    # Advanced cleaning
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    # Enhanced stopwords
    enhanced_stopwords = set(STOPWORDS)
    enhanced_stopwords.update([
        'video', 'youtube', 'comment', 'like', 'channel', 'subscribe', 'watch',
        'really', 'just', 'get', 'one', 'go', 'know', 'think', 'see', 'good',
        'great', 'thanks', 'time', 'make', 'people', 'way', 'much', 'even',
        'also', 'well', 'want', 'come', 'new', 'first', 'best', 'love'
    ])
    
    if not text.strip():
        st.info(f"No meaningful text for {sentiment_type.lower()} word cloud")
        return
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='rgba(0,0,0,0)',
        mode='RGBA',
        colormap='viridis' if sentiment_type == 'POSITIVE' else 'plasma',
        stopwords=enhanced_stopwords,
        max_words=100,
        relative_scaling=0.5,
        collocations=False,
        min_font_size=10,
        prefer_horizontal=0.9
    ).generate(text)
    
    # Create matplotlib figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'{sentiment_type.title()} Comments Word Cloud', 
                color='white', fontsize=18, pad=20, weight='bold')
    
    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    st.pyplot(fig, use_container_width=True, bbox_inches='tight', 
              facecolor='black', edgecolor='white')
    plt.close()

def create_advanced_filters():
    """Advanced filtering interface with multiple criteria"""
    if st.session_state.processed_df is None or st.session_state.processed_df.empty:
        return None
    
    df = st.session_state.processed_df
    
    st.markdown("### 🔍 Advanced Comment Filters")
    
    # Create filter columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_filter = st.multiselect(
            "Sentiment",
            options=df['Sentiment'].unique().tolist(),
            default=df['Sentiment'].unique().tolist(),
            key="sentiment_multi"
        )
    
    with col2:
        if 'Intent' in df.columns:
            intent_options = [x for x in df['Intent'].unique() if x != 'N/A']
            intent_filter = st.multiselect(
                "Intent",
                options=intent_options,
                default=intent_options,
                key="intent_multi"
            )
        else:
            intent_filter = []
    
    with col3:
        score_range = st.slider(
            "Confidence Score Range",
            min_value=float(df['Score'].min()),
            max_value=float(df['Score'].max()),
            value=(float(df['Score'].min()), float(df['Score'].max())),
            step=0.01,
            key="score_slider"
        )
    
    with col4:
        df['Word_Count'] = df['Comment'].str.split().str.len()
        word_range = st.slider(
            "Word Count Range",
            min_value=int(df['Word_Count'].min()),
            max_value=int(df['Word_Count'].max()),
            value=(int(df['Word_Count'].min()), int(df['Word_Count'].max())),
            key="word_slider"
        )
    
    # Additional filters row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        keyword_filter = st.text_input(
            "Keyword Search",
            placeholder="Enter keyword to search in comments",
            key="keyword_input"
        )
    
    with col6:
        sort_by = st.selectbox(
            "Sort By",
            options=['Score', 'Word_Count', 'Comment_Length'],
            key="sort_select"
        )
    
    with col7:
        sort_order = st.radio(
            "Sort Order",
            options=['Descending', 'Ascending'],
            horizontal=True,
            key="sort_order"
        )
    
    with col8:
        show_count = st.number_input(
            "Show Comments",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="show_count"
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply sentiment filter
    if sentiment_filter:
        filtered_df = filtered_df[filtered_df['Sentiment'].isin(sentiment_filter)]
    
    # Apply intent filter
    if intent_filter and 'Intent' in df.columns:
        filtered_df = filtered_df[filtered_df['Intent'].isin(intent_filter)]
    
    # Apply score range filter
    filtered_df = filtered_df[
        (filtered_df['Score'] >= score_range[0]) & 
        (filtered_df['Score'] <= score_range[1])
    ]
    
    # Apply word count filter
    filtered_df = filtered_df[
        (filtered_df['Word_Count'] >= word_range[0]) & 
        (filtered_df['Word_Count'] <= word_range[1])
    ]
    
    # Apply keyword filter
    if keyword_filter:
        filtered_df = filtered_df[
            filtered_df['Comment'].str.contains(keyword_filter, case=False, na=False)
        ]
    
    # Sort results
    ascending = sort_order == 'Ascending'
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    # Display filter summary
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    with col_summary1:
        st.metric("Original Comments", len(df))
    with col_summary2:
        st.metric("Filtered Comments", len(filtered_df))
    with col_summary3:
        filter_pct = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
        st.metric("Filter Efficiency", f"{filter_pct:.1f}%")
    
    return filtered_df.head(show_count)

def display_filtered_comments(filtered_df):
    """Display filtered comments in an elegant format"""
    if filtered_df is None or filtered_df.empty:
        st.info("No comments match the current filters")
        return
    
    st.markdown("### 📝 Filtered Comments")
    
    for idx, row in filtered_df.iterrows():
        # Create sentiment emoji mapping
        sentiment_emojis = {
            'POSITIVE': '😊',
            'NEUTRAL': '😐', 
            'NEGATIVE': '😞'
        }
        
        # Create expandable comment cards
        with st.expander(
            f"{sentiment_emojis.get(row['Sentiment'], '❓')} "
            f"{row['Sentiment']} - Score: {row['Score']:.3f} "
            f"({row['Word_Count']} words)"
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Comment:**")
                st.write(row['Comment'])
            
            with col2:
                st.markdown("**Metrics:**")
                st.metric("Sentiment", row['Sentiment'])
                st.metric("Confidence", f"{row['Score']:.3f}")
                if 'Intent' in row and row['Intent'] != 'N/A':
                    st.metric("Intent", row['Intent'])
                st.metric("Length", f"{row['Word_Count']} words")

def generate_insights():
    """Generate intelligent insights from the analysis"""
    if st.session_state.processed_df is None or st.session_state.processed_df.empty:
        return
    
    df = st.session_state.processed_df
    
    st.markdown("### 🧠 AI-Generated Insights")
    
    insights = []
    
    # Sentiment insights
    sentiment_counts = df['Sentiment'].value_counts()
    dominant_sentiment = sentiment_counts.index[0]
    dominance_pct = sentiment_counts.iloc[0] / len(df) * 100
    
    insights.append({
        'title': 'Sentiment Dominance',
        'content': f"**{dominant_sentiment.title()}** sentiment dominates with {dominance_pct:.1f}% of comments",
        'type': 'sentiment'
    })
    
    # Confidence insights
    avg_confidence = df['Score'].mean()
    high_confidence = len(df[df['Score'] > 0.8])
    
    insights.append({
        'title': 'Model Confidence',
        'content': f"Average confidence score is **{avg_confidence:.3f}** with {high_confidence} high-confidence predictions",
        'type': 'confidence'
    })
    
    # Length insights
    df['Word_Count'] = df['Comment'].str.split().str.len()
    avg_length = df['Word_Count'].mean()
    long_comments = len(df[df['Word_Count'] > 20])
    
    insights.append({
        'title': 'Comment Engagement',
        'content': f"Average comment length is **{avg_length:.1f} words** with {long_comments} detailed comments (>20 words)",
        'type': 'engagement'
    })
    
    # Sentiment-length correlation
    pos_avg_length = df[df['Sentiment'] == 'POSITIVE']['Word_Count'].mean()
    neg_avg_length = df[df['Sentiment'] == 'NEGATIVE']['Word_Count'].mean()
    
    if pos_avg_length > neg_avg_length * 1.2:
        engagement_insight = "Positive comments tend to be more detailed and engaged"
    elif neg_avg_length > pos_avg_length * 1.2:
        engagement_insight = "Negative comments show higher engagement with longer text"
    else:
        engagement_insight = "Comment length is consistent across sentiment types"
    
    insights.append({
        'title': 'Engagement Pattern',
        'content': engagement_insight,
        'type': 'pattern'
    })
    
    # Intent insights
    if 'Intent' in df.columns:
        top_intent = df['Intent'].value_counts().index[0]
        intent_pct = df['Intent'].value_counts().iloc[0] / len(df) * 100
        insights.append({
            'title': 'Primary Intent',
            'content': f"Most common intent is **{top_intent}** ({intent_pct:.1f}% of comments)",
            'type': 'intent'
        })
    
    # Display insights in cards
    for i, insight in enumerate(insights):
        st.markdown(f"""
        <div class="insight-card">
            <h4>💡 {insight['title']}</h4>
            <p>{insight['content']}</p>
        </div>
        """, unsafe_allow_html=True)

def analyze_video(video_id):
    """Enhanced video analysis with comprehensive error handling"""
    if not video_id:
        st.error("No video ID provided")
        return False
    
    fetch_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Create progress tracking
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch comments
            status_text.text("🔍 Fetching comments from video...")
            progress_bar.progress(20)
            
            comments = cached_fetch_comments(fetch_url, max_comments=500)
            
            if not comments:
                st.error("❌ Failed to fetch comments. Please check if the video exists and has comments enabled.")
                progress_bar.empty()
                status_text.empty()
                return False
            
            # Step 2: Analyze sentiment and intent
            status_text.text("🧠 Analyzing sentiment and intent...")
            progress_bar.progress(60)
            
            analysis_result = cached_analyze_data(comments)
            
            if len(analysis_result) != 4:
                st.error(f"❌ Analysis returned unexpected format: {analysis_result}")
                progress_bar.empty()
                status_text.empty()
                return False
            
            sentiment_summary, sentiment_detailed, intent_summary, intent_detailed = analysis_result
            
            # Validate analysis results
            if not isinstance(sentiment_summary, dict):
                st.error(f"❌ Invalid sentiment summary format: {type(sentiment_summary)}")
                progress_bar.empty()
                status_text.empty()
                return False
            
            if not isinstance(sentiment_detailed, list):
                st.error(f"❌ Invalid sentiment detailed format: {type(sentiment_detailed)}")
                progress_bar.empty()
                status_text.empty()
                return False
            
            # Step 3: Process data
            status_text.text("📊 Processing analysis results...")
            progress_bar.progress(80)
            
            # Create enhanced DataFrame
            df_data = []
            for i, comment in enumerate(comments):
                # Handle sentiment data
                if i < len(sentiment_detailed):
                    sentiment_info = sentiment_detailed[i]
                    if isinstance(sentiment_info, dict):
                        sentiment_label = sentiment_info.get('label', 'UNKNOWN')
                        sentiment_score = sentiment_info.get('score', 0.0)
                    elif isinstance(sentiment_info, tuple) and len(sentiment_info) >= 3:
                        sentiment_label = sentiment_info[1]
                        sentiment_score = sentiment_info[2]
                    else:
                        sentiment_label = 'UNKNOWN'
                        sentiment_score = 0.0
                else:
                    sentiment_label = 'UNKNOWN'
                    sentiment_score = 0.0
                
                # Handle intent data
                if i < len(intent_detailed):
                    intent_info = intent_detailed[i]
                    if isinstance(intent_info, dict):
                        intent_label = intent_info.get('label', 'N/A')
                    elif isinstance(intent_info, tuple) and len(intent_info) >= 2:
                        intent_label = intent_info[1]
                    else:
                        intent_label = 'N/A'
                else:
                    intent_label = 'N/A'
                
                df_data.append({
                    'Comment': str(comment),
                    'Sentiment': sentiment_label,
                    'Score': float(sentiment_score),
                    'Intent': intent_label,
                    'Comment_Length': len(str(comment))
                })
            
            if not df_data:
                st.error("❌ No data processed successfully")
                progress_bar.empty()
                status_text.empty()
                return False
            
            df = pd.DataFrame(df_data)
            df['Word_Count'] = df['Comment'].str.split().str.len()
            
            # Step 4: Store results
            status_text.text("💾 Saving analysis results...")
            progress_bar.progress(100)
            
            st.session_state.comments_data = comments
            st.session_state.sentiment_data = {
                'summary': sentiment_summary,
                'detailed': sentiment_detailed
            }
            st.session_state.intent_data = {
                'summary': intent_summary, 
                'detailed': intent_detailed
            }
            st.session_state.processed_df = df
            st.session_state.current_video_id = video_id
            st.session_state.analysis_metadata = {
                'total_comments': len(comments),
                'analysis_timestamp': pd.Timestamp.now(),
                'video_id': video_id
            }
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Critical error in analysis: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            
            progress_bar.empty()
            status_text.empty()
            return False

def display_results():
    """Enhanced results display with beautiful layout"""
    # Validate data before displaying
    if (not st.session_state.sentiment_data or 
        not st.session_state.intent_data or 
        st.session_state.processed_df is None or 
        st.session_state.processed_df.empty):
        st.error("❌ No valid analysis data available to display")
        return
    
    # Header with gradient background
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            🎬 Analysis Results Dashboard
        </h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Comprehensive YouTube Comment Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced metrics
    create_enhanced_metrics()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🧠 Intent Analysis", 
        "📈 Trends & Patterns", 
        "☁️ Text Insights", 
        "🔍 Advanced Explorer"
    ])
    
    with tab1:
        st.markdown("### Sentiment Distribution Analysis")
        plot_enhanced_sentiment_distribution()
        
        st.markdown("### Comprehensive Score Analysis") 
        plot_sentiment_score_analysis()
    
    with tab2:
        st.markdown("### Intent Classification Results")
        plot_enhanced_intent_analysis()
        
        # Intent-Sentiment correlation
        if st.session_state.processed_df is not None and 'Intent' in st.session_state.processed_df.columns:
            st.markdown("### Intent-Sentiment Correlation")
            df = st.session_state.processed_df
            
            # Filter out N/A intents
            df_filtered = df[df['Intent'] != 'N/A']
            
            if not df_filtered.empty:
                correlation_data = pd.crosstab(df_filtered['Intent'], df_filtered['Sentiment'], normalize='index') * 100
                
                fig = px.imshow(
                    correlation_data.values,
                    x=correlation_data.columns,
                    y=correlation_data.index,
                    color_continuous_scale='RdYlBu_r',
                    aspect='auto',
                    title="Intent vs Sentiment Heatmap (%)"
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No meaningful intent data available for correlation analysis")
    
    with tab3:
        st.markdown("### Sentiment Trend Analysis")
        plot_sentiment_trend_analysis()
        
        # Additional trend metrics
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            col1, col2, col3 = st.columns(3)
            
            with col1:
                volatility = df['Score'].std()
                st.metric("Score Volatility", f"{volatility:.3f}")
            
            with col2:
                pos_trend = len(df[df['Sentiment'] == 'POSITIVE']) / len(df) * 100
                st.metric("Positive Trend", f"{pos_trend:.1f}%")
            
            with col3:
                avg_confidence = df['Score'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with tab4:
        st.markdown("### Word Cloud Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Positive Comments")
            generate_enhanced_wordcloud('POSITIVE')
        
        with col2:
            st.markdown("#### Negative Comments") 
            generate_enhanced_wordcloud('NEGATIVE')
        
        # Word frequency analysis
        st.markdown("### Word Frequency Analysis")
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            
            # Create word frequency comparison
            positive_text = ' '.join(df[df['Sentiment'] == 'POSITIVE']['Comment'])
            negative_text = ' '.join(df[df['Sentiment'] == 'NEGATIVE']['Comment'])
            
            # Process and count words
            def get_word_freq(text, n=15):
                words = re.findall(r'\b\w+\b', text.lower())
                stop_words = set(STOPWORDS) | {'video', 'youtube', 'like', 'really', 'just'}
                words = [w for w in words if w not in stop_words and len(w) > 2]
                return Counter(words).most_common(n)
            
            pos_words = get_word_freq(positive_text)
            neg_words = get_word_freq(negative_text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if pos_words:
                    pos_df = pd.DataFrame(pos_words, columns=['Word', 'Count'])
                    fig = px.bar(pos_df, x='Count', y='Word', orientation='h',
                               title="Top Positive Words", color_discrete_sequence=['#2ecc71'])
                    fig.update_layout(template="plotly_dark", height=400,
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if neg_words:
                    neg_df = pd.DataFrame(neg_words, columns=['Word', 'Count'])
                    fig = px.bar(neg_df, x='Count', y='Word', orientation='h',
                               title="Top Negative Words", color_discrete_sequence=['#e74c3c'])
                    fig.update_layout(template="plotly_dark", height=400,
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### Advanced Comment Explorer")
        
        # Advanced filters
        filtered_df = create_advanced_filters()
        
        st.markdown("---")
        
        # Display filtered comments
        if filtered_df is not None and not filtered_df.empty:
            display_filtered_comments(filtered_df)
        
        st.markdown("---")
        
        # Export functionality
        if st.session_state.processed_df is not None:
            st.markdown("### 📥 Export Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Full Dataset"):
                    csv = st.session_state.processed_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"youtube_analysis_{st.session_state.current_video_id}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if filtered_df is not None and st.button("📋 Export Filtered Data"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Filtered CSV",
                        data=csv,
                        file_name=f"youtube_filtered_{st.session_state.current_video_id}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("📈 Export Summary"):
                    summary_data = {
                        'Total Comments': len(st.session_state.processed_df),
                        'Sentiment Distribution': st.session_state.sentiment_data['summary'],
                        'Intent Distribution': st.session_state.intent_data['summary'],
                        'Average Score': st.session_state.processed_df['Score'].mean(),
                        'Analysis Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    st.json(summary_data)
    
    # AI Insights at the bottom
    st.markdown("---")
    generate_insights()
    
    # Success message with animation
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
                padding: 1rem; border-radius: 10px; text-align: center; margin-top: 2rem;">
        <h3 style="color: white; margin: 0;">🎉 Analysis Complete!</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Powered by advanced NLP models with comprehensive sentiment and intent analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application with enhanced error handling"""
    try:
        # Get video_id from query parameters (matching what app.py sends)
        params = st.query_params
        video_id = (
            params.get("video_id", None) or 
            params.get("video_url", None) or  # fallback
            os.getenv("VIDEO_URL")
        )
        
        if video_id:
            video_id = video_id.strip()
            
            # If we received a full URL instead of just ID, extract the ID
            if "youtube.com" in video_id or "youtu.be" in video_id:
                video_id = extract_video_id(video_id)
            
            if not video_id:
                st.error("❌ Could not extract valid video ID")
                return
            
            # Check if already analyzed
            if (st.session_state.current_video_id == video_id and 
                st.session_state.sentiment_data is not None):
                display_results()
            else:
                # New analysis needed
                if st.session_state.current_video_id != video_id:
                    # Clear previous data
                    for key in ['comments_data', 'sentiment_data', 'intent_data', 
                               'processed_df', 'current_video_id', 'analysis_metadata']:
                        st.session_state[key] = None
                
                # Show loading screen
                with st.container():
                    st.markdown("""
                    <div style="text-align: center; padding: 3rem;">
                        <h2 style="color: #f8f8f2;">🎬 Initializing Analysis...</h2>
                        <p style="color: #6272a4;">Processing video ID: <code>{}</code></p>
                    </div>
                    """.format(video_id), unsafe_allow_html=True)
                    
                    if analyze_video(video_id):
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please check the video ID and try again.")
        else:
            # Show welcome screen
            st.markdown("""
            <div style="text-align: center; padding: 4rem; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 20px; margin: 2rem 0;">
                <h1 style="color: white; margin-bottom: 1rem; font-size: 3rem;">
                    🎬 YouTube Analytics Dashboard
                </h1>
                <p style="color: rgba(255,255,255,0.9); font-size: 1.3rem; margin-bottom: 2rem;">
                    Advanced Sentiment & Intent Analysis Platform
                </p>
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; 
                            border-radius: 15px; margin-top: 2rem;">
                    <h3 style="color: white; margin-bottom: 1rem;">🚀 Launch from Gradio App</h3>
                    <p style="color: rgba(255,255,255,0.8);">
                        Please use the Gradio interface to select a video and launch this dashboard
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature showcase
            col1, col2, col3 = st.columns(3)
            
            features = [
                {
                    'icon': '🧠',
                    'title': 'Advanced NLP',
                    'description': 'Deep sentiment analysis with confidence scoring and intent classification'
                },
                {
                    'icon': '📊', 
                    'title': 'Interactive Viz',
                    'description': 'Beautiful charts, trends, and word clouds with real-time filtering'
                },
                {
                    'icon': '🔍',
                    'title': 'Smart Insights',
                    'description': 'AI-generated insights and advanced filtering capabilities'
                }
            ]
            
            for i, feature in enumerate(features):
                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div style="background: rgba(30, 30, 46, 0.8); padding: 2rem; 
                                border-radius: 15px; border: 1px solid #44475a; 
                                text-align: center; height: 200px;">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">{feature['icon']}</div>
                        <h3 style="color: #f8f8f2; margin-bottom: 1rem;">{feature['title']}</h3>
                        <p style="color: #6272a4;">{feature['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    main()
