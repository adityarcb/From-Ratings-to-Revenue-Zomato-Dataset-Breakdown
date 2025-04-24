# === Import Libraries ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import folium

# === Load and Clean Data ===
file_path = "/content/Zomato Dataset 2.xlsx"  # Adjusted for uploaded file
df = pd.read_excel(file_path, sheet_name='Zomato Dataset')

# Drop unnamed columns
df_cleaned = df.loc[:, ~df.columns.str.contains('^Unnamed')].copy()

# Drop rows with essential missing values
df_cleaned.dropna(subset=[
    'Aggregate rating', 'Votes', 'Cuisines', 'Average Cost for two',
    'Price range', 'Longitude', 'Latitude'
], inplace=True)

# Remove outliers in Votes (top 1%)
df_cleaned = df_cleaned[df_cleaned['Votes'] < df_cleaned['Votes'].quantile(0.99)]


# === Objective 1: Analyze Restaurant Ratings and Popularity ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cleaned, x='Aggregate rating', y='Votes')
plt.title('Relationship between Aggregate Rating and Votes')
plt.xlabel('Aggregate Rating')
plt.ylabel('Votes')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Objective 3: Cuisine Preferences Across Countries ===

# --- Top cuisines by frequency in top 3 countries ---
top_countries = df_cleaned['Country Code'].value_counts().head(3).index
cuisine_data = df_cleaned[df_cleaned['Country Code'].isin(top_countries)]

cuisine_counts = cuisine_data['Cuisines'].str.split(',').explode().str.strip().value_counts().head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index)
plt.title('Top 20 Cuisines Across Top Countries')
plt.xlabel('Count')
plt.ylabel('Cuisine')
plt.tight_layout()
plt.show()

# === Objective 4: Predict Restaurant Ratings Using Machine Learning ===
# --- Correlation heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(df_cleaned[['Aggregate rating', 'Votes', 'Price range', 'Average Cost for two']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# --- Prepare data for ML ---
df_ml = df_cleaned.copy()
df_ml['Has Table booking'] = df_ml['Has Table booking'].map({'Yes': 1, 'No': 0})
df_ml['Has Online delivery'] = df_ml['Has Online delivery'].map({'Yes': 1, 'No': 0})
df_ml['Cuisine Count'] = df_ml['Cuisines'].apply(lambda x: len(str(x).split(',')))

features = ['Price range', 'Average Cost for two', 'Has Table booking',
            'Has Online delivery', 'Votes', 'Cuisine Count']
X = df_ml[features]
y = df_ml['Aggregate rating']

# --- Feature scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Model training and evaluation ---
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# === Objective 5: Geographical Analysis of Restaurant Distribution ===
import branca.colormap as cm
from folium.plugins import MarkerCluster

# ... (Rest of your code) ...
map_sample = df.sample(n=1000, random_state=42)  # Adjust n for desired sample size
min_rating = map_sample['Aggregate rating'].min()
max_rating = map_sample['Aggregate rating'].max()
colormap = cm.linear.RdYlGn_09.scale(min_rating, max_rating)
colormap.caption = 'Aggregate Rating Scale'

map_obj = folium.Map(location=[map_sample['Latitude'].mean(), map_sample['Longitude'].mean()], zoom_start=2)
marker_cluster = MarkerCluster().add_to(map_obj)

for _, row in map_sample.iterrows():
    rating = row['Aggregate rating']
    radius = row['Votes'] / map_sample['Votes'].max() * 10
    popup_text = f"""
    <b>{row['Restaurant Name']}</b><br>
    Rating: {rating}<br>
    Votes: {row['Votes']}<br>
    Cuisine: {row['Cuisines']}
    """
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=radius,
        color='black',  # Stroke color
        weight=1,  # Stroke width
        fill=True,
        fill_color=colormap(rating),
        fill_opacity=0.8,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(marker_cluster)

colormap.add_to(map_obj)
display(map_obj)
