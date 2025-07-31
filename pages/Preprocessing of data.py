import streamlit as st

# Page config
st.set_page_config(
    page_title="Preprocessing of data",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style
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
st.title("Preprocessing of data")

st.header("➤ Preprocessing: raw_sample.csv")

st.subheader("Dropping Unnecessary Columns")

st.markdown("""
The original `raw_sample.csv` file contains click log data, with two binary columns — `clk` and `nonclk` — representing whether an ad was clicked or not.

To simplify this, we create a single `clicked` column where:
- `clicked = 1` means the ad was clicked
- `clicked = 0` means the ad was not clicked

We then drop the original `clk` and `nonclk` columns.
""")

st.code("""
import pandas as pd

df = pd.read_csv("raw_sample.csv")
df["clicked"] = df["clk"]
df.drop(columns=["clk", "nonclk"], inplace=True)
""", language="python")

st.markdown("""
This results in a cleaner and more interpretable dataset with the following columns:

`user`, `time_stamp`, `adgroup_id`, `pid`, `clicked`

This format makes downstream analysis and modeling easier and avoids column redundancy.
""")

st.subheader("Time Feature Extraction")

st.markdown("""
We convert the Unix `time_stamp` column into a readable datetime format and extract:
- `hour`: Hour of interaction
- `weekday`: Day of week (0 = Monday)

These features are useful for plotting **engagement patterns** across time and identifying when users are most active.
""")

st.code("""
raw_sample_df["datetime"] = pd.to_datetime(raw_sample_df["time_stamp"], unit="s")
raw_sample_df["hour"] = raw_sample_df["datetime"].dt.hour
raw_sample_df["weekday"] = raw_sample_df["datetime"].dt.weekday
""", language="python")

st.subheader("Checking for Missing Values")

st.markdown("""
Before finalizing the cleaned dataset, we perform a check for any **null or missing values** across columns:
""")

st.code("""raw_sample_df.isnull().sum()""")

st.markdown("""
There are no missing values in the dataset, so no further action is required at this stage.
""")

st.subheader("Memory & Data Type Check")

st.markdown("""
To understand the dataset size and data types, we use:
""")

st.code("""raw_sample_df.info()""")

st.markdown("""
The output shows:
`dtypes: datetime64[ns](1), int32(2), int64(3), object(1)
memory usage: 1.2+ GB`\n
The memory usage is significant, which can slow down processing. To improve performance, we convert large numeric columns to more memory-efficient types.
""")

st.code("""
for col in raw_sample_df.select_dtypes(include=['int64']).columns:
raw_sample_df[col] = pd.to_numeric(raw_sample_df[col], downcast='unsigned')

for col in raw_sample_df.select_dtypes(include=['float64']).columns:
raw_sample_df[col] = pd.to_numeric(raw_sample_df[col], downcast='float')

raw_sample_df["pid"] = raw_sample_df["pid"].astype("category")

raw_sample_df.info()
""")

st.markdown("""
After optimization, the dataset info shows: 
`dtypes: category(1), datetime64[ns](1), int32(2), uint32(2), uint8(1)
memory usage: 658.5 MB`\n
By converting columns to appropriate data types (such as `int64` to `uint8`, and `object` to `category`), we were able to reduce the dataset's memory usage from over **1.2 GB** to just **658.5 MB**. This not only minimizes the storage footprint but also leads to faster execution of operations like filtering, grouping, and merging. Efficient memory usage is especially important when working with large-scale data, as it improves processing speed, scalability, and compatibility with data analysis libraries such as pandas.
""")

