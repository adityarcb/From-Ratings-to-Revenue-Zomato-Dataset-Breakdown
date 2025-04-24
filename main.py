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
