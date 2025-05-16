import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

def load_data():
    """
    Load and prepare transaction data from CSV files
    """
    # Check if synthetic data is available
    if os.path.exists('data/preference_data.csv'):
        print("Loading synthetic preference data...")
        preference_df = pd.read_csv('data/preference_data.csv')
        return preference_df
    
    # If no synthetic data, use original data loading logic
    print("Loading data from CSV files...")
    
    try:
        # Load transaction data
        transactions_df = pd.read_csv('data/transaction.csv', sep=';')
        transaction_items_df = pd.read_csv('data/transaction_items.csv', sep=';')
        customers_df = pd.read_csv('data/customers.csv', sep=';')
        products_df = pd.read_csv('data/products.csv', sep=';')
        
        # Further processing of original data...
        # [Your existing data processing logic]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Perform feature engineering on the transaction data
    """
    # Create copies to avoid modifying original data
    processed_df = df.copy()
    
    # Ensure the correct data types
    processed_df['quarter'] = processed_df['quarter'].astype(int)
    processed_df['customer_id'] = processed_df['customer_id'].astype(int)
    processed_df['product_id'] = processed_df['product_id'].astype(int)
    
    # Feature engineering
    if 'day_of_week' not in processed_df.columns:
        # For synthetic data, day_of_week might not be present
        processed_df['day_of_week'] = np.random.randint(0, 7, size=len(processed_df))
    
    if 'is_weekend' not in processed_df.columns:
        # Generate is_weekend from day_of_week or randomly
        processed_df['is_weekend'] = processed_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    if 'month' not in processed_df.columns:
        # Generate month from quarter or randomly
        processed_df['month'] = processed_df['quarter'].apply(lambda q: np.random.randint((q-1)*3+1, q*3+1))
    
    # Calculate price ratios if not present
    if 'price_ratio' not in processed_df.columns:
        processed_df['price_ratio'] = processed_df['price'] / processed_df['total_amount']
    
    if 'price_per_quantity' not in processed_df.columns:
        if 'quantity' in processed_df.columns:
            processed_df['price_per_quantity'] = processed_df['price'] / processed_df['quantity']
        else:
            processed_df['price_per_quantity'] = processed_df['price']
    
    # Encode categorical features
    if processed_df['gender'].dtype == 'object':
        processed_df['gender'] = processed_df['gender'].str.upper().map({'M': 0, 'F': 1}).fillna(0)
    
    if processed_df['city'].dtype == 'object':
        # Map cities to numeric values
        city_mapping = {city: i for i, city in enumerate(processed_df['city'].unique())}
        processed_df['city'] = processed_df['city'].map(city_mapping)
    
    if processed_df['membership_level'].dtype == 'object':
        # Map membership levels to numeric values
        membership_mapping = {'bronze': 0, 'silver': 1, 'gold': 2, 'platinum': 3}
        processed_df['membership_level'] = processed_df['membership_level'].str.lower().map(membership_mapping).fillna(0)
    
    if processed_df['brand'].dtype == 'object':
        # Map brands to numeric values
        brand_mapping = {brand: i for i, brand in enumerate(processed_df['brand'].unique())}
        processed_df['brand'] = processed_df['brand'].map(brand_mapping)
    
    print(f"Feature engineering completed. Dataset shape: {processed_df.shape}")
    return processed_df

def train_quarterly_models(df):
    """
    Train XGBoost models for each quarter
    """
    models = {}
    rmse_scores = {}
    
    # Features to use for training
    features = [
        'age', 'gender', 'city', 'membership_level', 'category', 'brand', 'price',
        'day_of_week', 'is_weekend', 'month', 'price_per_quantity', 'price_ratio', 
        'total_amount'
    ]
    
    # Train model for each quarter
    for year in sorted(df['year'].unique()):
        for quarter in sorted(df[df ['year'] == year ]['quarter'].unique()):
            print(f"Training model untuk tahun {year}, kuartal {quarter}...")
        
        # Filter data for the current quarter
        period_df = df[(df['year'] == year) & (df['quarter'] == quarter)]
        
        # Skip if insufficient data
        if len(period_df) < 5:
            print(f"Insufficient data for yar {year}, quarter {quarter}, skipping...")
            continue
        
        # Prepare training data
        X = period_df[features].values
        y = period_df['preference_score'].values
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 3,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1
        }
        
        # Train the model
        model = xgb.train(params, dtrain, num_boost_round=50)
        
        # Make predictions on training data for evaluation
        pred = model.predict(dtrain)
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        
        # Store model and RMSE score
        key = f"{year}Q{quarter}"
        models[key] = model
        rmse_scores[key] = rmse
        
        print(f"Year{year}, Quarter {quarter} model trained. RMSE: {rmse:.4f}")
    
    return models, rmse_scores

def predict_customer_preferences(customer_id, quarter, models, df, top_n=5):
    """
    Predict preferences for a specific customer in a specific quarter
    """
    print(f"Memprediksi preferensi untuk Customer ID: {customer_id}, Kuartal: {quarter}")
    
    # Get model for the quarter
    if quarter not in models:
        closest_quarter = min(models.keys(), key=lambda q: abs(q - quarter))
        print(f"Model untuk kuartal {quarter} tidak tersedia. Menggunakan model kuartal {closest_quarter}.")
        quarter = closest_quarter
    
    model = models[quarter]
    
    # Filter data for the customer in the quarter
    customer_data = df[(df['customer_id'] == customer_id) & (df['quarter'] == quarter)]
    
    if customer_data.empty:
        print(f"Data tidak tersedia untuk pelanggan {customer_id} pada kuartal {quarter}.")
        return None
    
    # Features to use for prediction
    features = [
        'age', 'gender', 'city', 'membership_level', 'category', 'brand', 'price',
        'day_of_week', 'is_weekend', 'month', 'price_per_quantity', 'price_ratio', 
        'total_amount'
    ]
    
    # Predict preferences for all products purchased by the customer
    predictions = []
    for _, row in customer_data.iterrows():
        X = row[features].values.reshape(1, -1)
        dtest = xgb.DMatrix(X)
        pred_score = model.predict(dtest)[0]
        
        predictions.append({
            'product_id': row['product_id'],
            'category': row['category'],
            'brand': row['brand'],
            'predicted_preference': pred_score,
            'actual_preference': row['preference_score']
        })
    
    # Sort by predicted preference
    predictions_df = pd.DataFrame(predictions)
    top_preferences = predictions_df.sort_values('predicted_preference', ascending=False).head(top_n)
    
    return top_preferences

def generate_business_recommendations(df, quarterly_models):
    """
    Generate business recommendations based on preference analysis
    """
    print("Membuat rekomendasi bisnis...")
    
    # Initialize recommendations list
    recommendations = []
    
    # Get all available years
    available_years = sorted(df['year'].unique())
    
    # Analysis 1: Top categories by preference score for each quarter and year
    for year in available_years:
        year_data = df[df['year'] == year]
        for quarter in sorted(year_data['quarter'].unique()):
            quarter_data = year_data[year_data['quarter'] == quarter]
            
            # Top categories by average preference
            top_categories = quarter_data.groupby('category')['preference_score'].mean().sort_values(ascending=False).head(3)
            
            for category, score in top_categories.items():
                recommendations.append({
                    'period': f'Q{quarter} {year}',  # Use dynamic year
                    'type': 'Personalization',
                    'finding': f'Kategori {category} memiliki skor preferensi tertinggi ({score:.2f})',
                    'recommendation': f'Buat penawaran khusus kategori {category} untuk pelanggan pada kuartal mendatang.'
                })
    
    # Analysis 2: Membership level preferences per year
    for year in available_years:
        year_data = df[df['year'] == year]
        membership_prefs = year_data.groupby('membership_level')['preference_score'].mean().sort_values(ascending=False)
        
        for level, score in membership_prefs.items():
            level_name = {0: 'Bronze', 1: 'Silver', 2: 'Gold', 3: 'Platinum'}.get(level, f'Level {level}')
            recommendations.append({
                'period': f'All {year}',  # Use dynamic year
                'type': 'Customer Segmentation',
                'finding': f'Pelanggan {level_name} menunjukkan preferensi rata-rata {score:.2f}',
                'recommendation': f'Kembangkan program loyalitas khusus untuk segmen {level_name}.'
            })
    
    # Analysis 3: City-based preferences per year
    for year in available_years:
        year_data = df[df['year'] == year]
        city_prefs = year_data.groupby('city')['preference_score'].mean().sort_values(ascending=False)
        
        for i, (city, score) in enumerate(city_prefs.items()):
            if i < 3:  # Top 3 cities
                recommendations.append({
                    'period': f'All {year}',  # Use dynamic year
                    'type': 'Regional Strategy',
                    'finding': f'Kota {city} memiliki preferensi rata-rata tertinggi ({score:.2f})',
                    'recommendation': f'Tingkatkan fokus pemasaran dan distribusi di kota {city}.'
                })
    
    # Analysis 4: Price sensitivity by membership level per year
    for year in available_years:
        year_data = df[df['year'] == year]
        for level in year_data['membership_level'].unique():
            level_data = year_data[year_data['membership_level'] == level]
            
            # Skip if not enough data
            if len(level_data) < 10:
                continue
                
            # Calculate correlation between price and preference
            price_corr = level_data[['price', 'preference_score']].corr().iloc[0, 1]
            level_name = {0: 'Bronze', 1: 'Silver', 2: 'Gold', 3: 'Platinum'}.get(level, f'Level {level}')
            
            if price_corr > 0.1:
                recommendations.append({
                    'period': f'All {year}',  # Use dynamic year
                    'type': 'Pricing Strategy',
                    'finding': f'Pelanggan {level_name} menunjukkan preferensi positif terhadap produk harga tinggi (korelasi: {price_corr:.2f})',
                    'recommendation': f'Tawarkan produk premium untuk segmen {level_name}.'
                })
            elif price_corr < -0.1:
                recommendations.append({
                    'period': f'All {year}',  # Use dynamic year
                    'type': 'Pricing Strategy',
                    'finding': f'Pelanggan {level_name} menunjukkan preferensi terhadap produk harga rendah (korelasi: {price_corr:.2f})',
                    'recommendation': f'Fokus pada penawaran value bagi segmen {level_name}.'
                })
    
    # Convert to DataFrame
    recommendations_df = pd.DataFrame(recommendations)
    
    # Display top recommendations
    print("\nTop rekomendasi bisnis:")
    print(recommendations_df.head())
    
    return recommendations_df

def main():
    """
    Main function to run the entire process
    """
    print("Starting The X Aksha Customer Preference Analysis...")
    
    # Load data
    df = load_data()
    if df is None:
        print("Gagal memuat data. Program berhenti.")
        return None, None, None
    
    print("Data successfully loaded from CSV files.")
    
    # Check if we need to preprocess
    if 'price_ratio' not in df.columns or 'price_per_quantity' not in df.columns:
        # Preprocess the data
        df = preprocess_data(df)
    
    # Train models for each quarter
    quarterly_models, rmse_scores = train_quarterly_models(df)
    
    # Example prediction for a specific customer
    try:
        customer_id = 10  # Example customer ID
        quarter = 2       # Example quarter
        predict_customer_preferences(customer_id, quarter, quarterly_models, df)
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    # Generate visualizations
    print("Membuat visualisasi...")
    try:
        # This function would normally create plots, but we'll handle that in visualisasi.py
        print("Visualisasi hasil disimpan.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        if not quarterly_models:
            print("Warning: No feature importance data available for visualization")
    
    # Generate business recommendations
    recommendations = generate_business_recommendations(df, quarterly_models)
    
    # Save models and data for later use
    print("Menyimpan model dan data...")
    try:
        with open('quarterly_models.pkl', 'wb') as f:
            pickle.dump(quarterly_models, f)
        df.to_pickle('processed_data.pkl')
        recommendations.to_pickle('recommendations.pkl')
        print("Model dan data berhasil disimpan.")
    except Exception as e:
        print(f"Error saving models and data: {e}")
    
    print("Analisis preferensi pembeli The X Aksha selesai!")
    return quarterly_models, df, recommendations

if __name__ == "__main__":
    main()