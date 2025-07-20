import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import numpy as np

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('hospital_data.csv')
    df.columns = df.columns.str.strip().str.lower()
    df['visit_date'] = pd.to_datetime(np.random.choice(
        pd.date_range('2022-01-01', '2023-12-31'), size=len(df)))
    df['month'] = df['visit_date'].dt.month
    df['season'] = df['visit_date'].dt.month % 12 // 3 + 1
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df['season'] = df['season'].map(season_map)
    return df

df = load_data()

st.title("🧑‍⚕️ Patient Data Analysis Dashboard")
st.markdown("Built with **Streamlit**, **Matplotlib**, **Seaborn**, and **Plotly**")

# Sidebar filter
st.sidebar.header("Filters")
selected_gender = st.sidebar.selectbox("Select Gender", options=['All'] + list(df['gender'].unique()))
if selected_gender != 'All':
    df = df[df['gender'] == selected_gender]

# Age Distribution
st.subheader("Age Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['age'], kde=True, bins=20, color='skyblue', ax=ax1)
st.pyplot(fig1)

# Gender Pie Chart
st.subheader("Gender Distribution")
gender_counts = df['gender'].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
ax2.axis('equal')
st.pyplot(fig2)

# Top Conditions
st.subheader("Top Diagnosed Conditions")
top_conditions = df['condition'].value_counts().head(10)
st.bar_chart(top_conditions)

# Word Cloud
st.subheader("Word Cloud of Conditions")
text = ' '.join(df['condition'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig3, ax3 = plt.subplots()
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis('off')
st.pyplot(fig3)

# Cost Distribution
st.subheader("Treatment Cost Distribution")
fig4, ax4 = plt.subplots()
sns.boxplot(x=df['cost'], color='orange', ax=ax4)
st.pyplot(fig4)

# Monthly Visit Trend (Plotly)
st.subheader("Monthly Visit Trend")
df['month_str'] = df['visit_date'].dt.to_period('M').astype(str)
monthly_visits = df['month_str'].value_counts().sort_index()
fig5 = px.line(x=monthly_visits.index, y=monthly_visits.values, labels={'x': 'Month', 'y': 'Number of Visits'})
st.plotly_chart(fig5)
