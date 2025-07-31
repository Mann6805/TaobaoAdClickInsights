import streamlit as st

# Page config
st.set_page_config(
    page_title="Dataset Description",
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
st.title("Dataset Description")

# Section: Source
st.header("➤ Dataset Source")
st.markdown("""
This project uses the **Alibaba Taobao Ad CTR Dataset** available from [Tianchi Lab](https://tianchi.aliyun.com/dataset/56#1).  
It contains user behavior logs, ad metadata, and user profile information collected over 8 days, and includes over **26 million records**.

This dataset is widely used for understanding user engagement and modeling **click-through rate (CTR)** in e-commerce advertising.
""")

# Section: Datasets Overview
st.header("➤ Available Datasets")

st.subheader("1. `raw_sample` – User Ad Interaction Logs")
st.markdown("""
This is the core behavioral dataset containing ad display and click records for 1.14 million users.

| Column       | Description                                     |
|--------------|-------------------------------------------------|
| `user`       | Anonymized user ID                              |
| `time_stamp` | Timestamp of the interaction (Unix format)      |
| `adgroup_id` | Ad ID that was shown to the user                |
| `pid`        | Scenario/placement ID                           |
| `noclk`      | 1 = not clicked, 0 = clicked                     |
| `clk`        | 1 = clicked, 0 = not clicked                     |
""")

st.markdown("""
The `raw_sample` table captures real user behavior — showing which ads were clicked, when, and in what context. 
While it tells us *what happened*, it doesn’t explain *why*. That’s where the other tables come in. 

By combining this behavioral data with ad details and user profiles, we can uncover deeper insights:
- Which users click more?
- Do certain ad types perform better?
- What time of day drives the most engagement?

This table is the starting point for understanding patterns in user interaction and click-through behavior.
""")


st.subheader("2. `ad_feature` – Ad Metadata")
st.markdown("""
Contains metadata for ads shown in `raw_sample`.

| Column         | Description                            |
|----------------|----------------------------------------|
| `adgroup_id`   | Ad ID (joins with `raw_sample`)        |
| `cate_id`      | Category of the product                |
| `campaign_id`  | Associated ad campaign ID              |
| `brand`        | Brand ID                               |
| `customer_id`  | Advertiser ID                          |
| `price`        | Price of the item                      |
""")

st.markdown("""
The `ad_feature` table tells us what kind of content was shown to users — including product category, price, brand, and campaign IDs.

On its own, it helps us understand the landscape of ads being served. When merged with clicks from `raw_sample`, 
we can analyze how different ad characteristics influence click behavior — such as:
- Do higher-priced items get clicked more?
- Are certain brands or categories more attractive?
- How do campaign strategies impact engagement?
""")

st.subheader("3. `user_profile` – User Demographics")
st.markdown("""
Demographic features of ~1 million users linked to the `raw_sample` interactions.

| Column                | Description                                      |
|-----------------------|--------------------------------------------------|
| `userid`              | User ID                                          |
| `cms_segid`           | Micro-segment ID                                 |
| `cms_group_id`        | Content group ID                                 |
| `final_gender_code`   | Gender (1 = male, 2 = female)                    |
| `age_level`           | Age bucket                                       |
| `pvalue_level`        | Consumption level (1 = low, 2 = mid, 3 = high)   |
| `shopping_level`      | Shopping depth (1 = shallow, 2 = moderate, 3 = deep) |
| `occupation`          | 1 = college student, 0 = not                     |
| `new_user_class_level`| City tier level                                  |
""")

st.markdown("""
The `user_profile` table gives us insight into the people behind the clicks — including gender, age group, shopping behavior, and city tier.

While click logs show actions, this table helps explain them. It enables segmentation and reveals:
- Which user segments are more likely to click?
- How do demographics affect ad engagement?
- Can spending level or shopping depth predict CTR?

This makes it a powerful tool for targeting and personalization.
""")
