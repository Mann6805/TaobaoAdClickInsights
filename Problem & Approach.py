import streamlit as st

# Page config
st.set_page_config(
    page_title="CTR Analysis - Problem & Approach",
    layout="wide"
)

# Theme
st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        color: #1f3a63;
    }
    h1, h2, h3 {
        color: #1f3a63;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Problem & Approach")

# Section: Problem Statement
st.header("➤ Problem Statement")
st.markdown("""
In today’s fast-paced e-commerce world, Click-Through Rate (CTR) has become one of the most important metrics for measuring how effectively ads are capturing user attention. It reflects not just user engagement, but also directly impacts how much revenue a platform can generate from advertising. When you’re serving millions of ads every day, even the slightest improvement in CTR prediction can lead to significant financial gains.

This project takes a deep dive into the factors that influence whether a user clicks on an ad or not. Using a large-scale dataset from Taobao, which includes detailed logs of user behavior, ad-related metadata, and demographic information, we aim to uncover patterns in user interaction. By understanding how different types of users respond to different kinds of ads at different times, we can build smarter models that predict CTR more accurately — ultimately helping platforms serve more relevant ads and improve the overall user experience.
""")

# Section: Why This Matters
st.header("➤ Why This Analysis is Important")
st.markdown("""
Understanding CTR behavior helps:
- Optimize ad placement and bidding strategies
- Improve user experience through personalized targeting
- Identify high-value user segments and conversion patterns
""")

# Section: Our Approach
st.header("➤ Our Approach")
st.markdown("""
To tackle this problem, we follow a structured pipeline:
1. **Preprocess** each dataset: user profile, ad features, and raw click logs
2. Perform **EDA on individual tables** to understand basic trends
3. **Merge** datasets to create a unified view combining user, ad, and behavioral data
4. Conduct **combined analysis** to uncover what truly drives clicks — from user demographics to ad price and context

This streamlined approach allows us to make sense of over 26 million records and extract meaningful insights for CTR optimization.
""")
