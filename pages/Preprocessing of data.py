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

st.header("➤ Preprocessing: ad_feature.csv")

st.markdown("""
We begin by loading the dataset and exploring its structure:""")

st.code("""
ad_features_df = pd.read_csv('ad_feature.csv')
ad_features_df.describe()
ad_features_df.info()
""")

st.markdown("""
This initial inspection reveals two important issues:

- The `price` column shows extreme values in its distribution, suggesting the presence of **outliers** that could skew the analysis.
- The `brand` column contains **missing values**, which will need to be addressed after handling price outliers.

We first address the outliers in `price` before deciding how to handle the null values in `brand`, to ensure that any imputation or filtering reflects the cleaned dataset.
""")

st.subheader("Removing Outliers from Price")

st.markdown("""
To better understand the distribution of ad prices, we use a boxplot. This allows us to clearly visualize extreme values (outliers) that fall outside the normal range of pricing.
""")

st.code("""
plt.figure(figsize=(10, 6))
sns.boxplot(
    x=ad_features_df['price'],
    color="#4C72B0",
    fliersize=4,
    flierprops=dict(
        marker='o',
        markerfacecolor='#E76F51',
        markeredgecolor='#E76F51',
        markersize=6,
        linestyle='none'
    )
)

plt.title('Boxplot of Price Distribution (With Outliers)', fontsize=14)
plt.xlabel('Price', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()

plt.show()
""")

st.markdown("""
This visualization confirms that the dataset contains unusually high price values, which could distort summary statistics.
""")

st.image("graphs/price_boxplot_outliers.png", caption="Boxplot of Price Distribution (With Outliers)", use_container_width =True)

st.markdown("""
To avoid distortion in pricing trends, we remove **extremely high and low values** from the `price` column using the **interquartile range (IQR)** method. This helps eliminate only the most extreme cases without discarding genuinely high prices that might reflect real market behavior.

The following code keeps prices between the 10th and 90th percentile, treating only extreme values beyond that as outliers:
""")

st.code("""
Q1 = ad_features_df['price'].quantile(0.10)
Q3 = ad_features_df['price'].quantile(0.90)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

ad_features_cleaned_df = ad_features_df[(ad_features_df['price'] >= lower_bound) & (ad_features_df['price'] <= upper_bound)]
ad_features_cleaned_df = ad_features_cleaned_df.reset_index(drop=True)
""", language="python")

st.markdown("""
Below is the updated boxplot after removing extreme outliers. While a few red points are still visible, they fall within a reasonable range and no longer represent the extreme values that were previously distorting the distribution.
""")

st.image("graphs/price_boxplot_without_outliers.png", caption="Boxplot of Price Distribution (Without Extreme Outliers)", use_container_width =True)

st.subheader("Optimizing Data Types Before Imputation")

st.markdown("""
Before filling missing values, we downcast the numeric columns to reduce memory usage and improve computational efficiency during similarity search:
""")

st.code("""
features = ['adgroup_id', 'cate_id', 'campaign_id', 'customer', 'price']

for col in features:
    ad_features_cleaned_df[col] = pd.to_numeric(ad_features_cleaned_df[col], downcast='unsigned')
""", language="python")

st.subheader("Filling Missing `brand` Values Using FAISS")

st.markdown("""
Instead of dropping rows with missing `brand`, we use FAISS (Facebook AI Similarity Search) to **predict the closest brand** based on similar ad features. This preserves more data and makes intelligent use of the surrounding structure.

The idea is to treat rows with known brand values as a training set and use Euclidean distance to find the nearest neighbor for each missing row:
""")

st.code("""
import faiss
import numpy as np
import pandas as pd

missing_brand_df = ad_features_cleaned_df[ad_features_cleaned_df['brand'].isnull()]
non_missing_brand_df = ad_features_cleaned_df[ad_features_cleaned_df['brand'].notnull()]

X = non_missing_brand_df[features].values
y = non_missing_brand_df['brand'].values
X_missing = missing_brand_df[features].values

X = X.astype(np.float32)
X_missing = X_missing.astype(np.float32)

index = faiss.IndexFlatL2(X.shape[1])
index.add(X)

batch_size = 10000
num_batches = len(X_missing) // batch_size + (1 if len(X_missing) % batch_size != 0 else 0)

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(X_missing))

    X_missing_batch = X_missing[start_idx:end_idx]
    distances, indices = index.search(X_missing_batch, 1)
    predicted_brands = y[indices.flatten()]

    ad_features_cleaned_df.loc[missing_brand_df.index[start_idx:end_idx], 'brand'] = predicted_brands

    print(f"Processed batch {i + 1}/{num_batches}")

print("Missing brand values filled using FAISS with Euclidean similarity.")
""", language="python")

st.markdown("""
FAISS is ideal for large-scale similarity search due to its speed and scalability.  
We use **Euclidean (L2) distance**, which performs well when feature vectors are numerical and normalized or of similar scale.

For more on FAISS and Euclidean similarity, refer to the official FAISS documentation:  
[https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
""")

st.markdown("""
After removing outliers and filling missing `brand` values, we now check the final structure and memory usage of the cleaned dataset: `dtypes: Int64(1), uint16(2), uint32(3)
memory usage: 19.3 MB`\n   
            
Through downcasting and optimized imputation, we've significantly reduced the dataset size — making it more efficient to store, load, and analyze at scale.
""")

st.header("➤ Preprocessing: user_profile.csv")

st.subheader("Downcasting Numeric Columns")

st.code("""
user_profile_df = pd.read_csv('user_profile.csv')
user_profile_df.columns = user_profile_df.columns.str.strip()

features = ['cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'shopping_level', 'occupation']

for col in features:
    user_profile_df[col] = pd.to_numeric(user_profile_df[col], downcast='unsigned')
""", language="python")

st.markdown("""
These features are used later to predict missing values, so we reduce their memory footprint early on.
""")

st.subheader("Imputing Missing Values Using FAISS")

st.markdown("""
We use **FAISS** to fill missing values in `pvalue_level` and `new_user_class_level` by finding the nearest neighbors based on other user attributes.

This preserves important records without dropping them and is effective at scale.
""")

st.code("""
import faiss
import numpy as np

columns_to_fill = ['pvalue_level', 'new_user_class_level']

for column in columns_to_fill:
    missing_df = user_profile_df[user_profile_df[column].isnull()]
    non_missing_df = user_profile_df[user_profile_df[column].notnull()]

    X = non_missing_df[features].values.astype(np.float32)
    y = non_missing_df[column].values
    X_missing = missing_df[features].values.astype(np.float32)

    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    batch_size = 10000
    num_batches = int(np.ceil(len(X_missing) / batch_size))
    missing_indices = missing_df.index.to_numpy()

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_missing))
        X_batch = X_missing[start:end]
        
        distances, indices = index.search(X_batch, k=1)
        predicted_vals = y[indices.flatten()]
        
        user_profile_df.loc[missing_indices[start:end], column] = predicted_vals

        print(f"Filled batch {i+1}/{num_batches} for column: {column}")
""", language="python")

st.markdown("""
Most columns in `user_profile.csv` are already categorical and well-structured, so no further transformation is necessary at this stage.
The final structure and memory usage of the cleaned dataset: `dtypes: float64(2), int64(1), uint8(6)
memory usage: 30.4 MB`\n   :
""")
