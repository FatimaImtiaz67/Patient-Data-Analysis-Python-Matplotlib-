import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np


# For better visuals
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
df = pd.read_csv('hospital data analysis.csv')

# Check basic info
print(df.info())

# Preview first few rows
df.head()

sns.histplot(data=df, x='Age', bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.show()

# Pie Chart
gender_counts = df['Gender'].value_counts()
gender_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'skyblue'])
plt.title('Gender Distribution')
plt.ylabel('')  # Remove y-axis label
plt.show()

# Bar Chart
sns.countplot(data=df, x='Gender', palette='pastel')
plt.title('Gender Count')
plt.xlabel('Gender')
plt.ylabel('Number of Patients')
plt.show()

# Count of visits by procedure
procedure_counts = df['Procedure'].value_counts().head(10)  # Top 10 procedures
sns.barplot(x=procedure_counts.values, y=procedure_counts.index, palette='Blues_r')
plt.title('Top 10 Procedures by Patient Visits')
plt.xlabel('Number of Patients')
plt.ylabel('Procedure')
plt.show()

condition_counts = df['Condition'].value_counts().head(10)
sns.barplot(x=condition_counts.values, y=condition_counts.index, palette='Greens_r')
plt.title('Top 10 Diagnosed Conditions')
plt.xlabel('Number of Patients')
plt.ylabel('Condition')
plt.show()


text = " ".join(str(cond) for cond in df['Condition'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Diagnosed Conditions')
plt.show()


# Generate random dates between Jan 2022 and Dec 2023
df['visit_date'] = pd.to_datetime(np.random.choice(pd.date_range('2022-01-01', '2023-12-31'), size=len(df)))

# Preview
df[['Patient_ID', 'visit_date']].head()
# Histogram of cost
sns.histplot(df['Cost'], bins=30, color='orange', kde=True)
plt.title('Distribution of Treatment Costs')
plt.xlabel('Cost')
plt.ylabel('Number of Patients')
plt.show()

# Box plot
sns.boxplot(x=df['Cost'], color='lightgreen')
plt.title('Treatment Cost Distribution (Box Plot)')
plt.xlabel('Cost')
plt.show()

avg_cost_by_procedure = df.groupby('Procedure')['Cost'].mean().sort_values(ascending=False).head(10)

sns.barplot(x=avg_cost_by_procedure.values, y=avg_cost_by_procedure.index, palette='coolwarm')
plt.title('Average Cost per Procedure (Top 10)')
plt.xlabel('Average Cost')
plt.ylabel('Procedure')
plt.show()

avg_cost_by_condition = df.groupby('Condition')['Cost'].mean().sort_values(ascending=False).head(10)

sns.barplot(x=avg_cost_by_condition.values, y=avg_cost_by_condition.index, palette='magma')
plt.title('Average Cost per Condition (Top 10)')
plt.xlabel('Average Cost')
plt.ylabel('Condition')
plt.show()


# Generate random visit dates between Jan 2022 and Dec 2023
df['visit_date'] = pd.to_datetime(np.random.choice(pd.date_range('2022-01-01', '2023-12-31'), size=len(df)))

# Extract month and season from visit_date
df['month'] = df['visit_date'].dt.month
df['month_name'] = df['visit_date'].dt.strftime('%B')
df['season'] = df['visit_date'].dt.month % 12 // 3 + 1  # Maps months to 1-4

season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
df['season'] = df['season'].map(season_map)

# Count of conditions per season
season_condition = df.groupby(['season', 'Condition']).size().unstack().fillna(0)

season_condition.T.plot(kind='line', figsize=(12, 6))
plt.title('Illness Trends by Season')
plt.xlabel('Condition')
plt.ylabel('Number of Cases')
plt.legend(title='Season')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

monthly_visits = df['visit_date'].dt.to_period('M').value_counts().sort_index()

monthly_visits.plot(kind='line', marker='o', color='teal')
plt.title('Patient Visit Trends Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Visits')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation heatmap of numerical columns
numeric_cols = df.select_dtypes(include='number')

sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()