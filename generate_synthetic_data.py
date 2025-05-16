import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_data(target_size=250, seed=42):
    """
    Generate synthetic data based on existing patterns to create a dataset
    with at least target_size transactions.
    
    Args:
        target_size: Minimum number of transaction records to generate
        seed: Random seed for reproducibility
        
    Returns:
        transactions_df, transaction_items_df: The generated DataFrames
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Load existing data
    base_dir = "data"
    customers_df = pd.read_csv(os.path.join(base_dir, "customers.csv"), sep=';')
    products_df = pd.read_csv(os.path.join(base_dir, "products.csv"), sep=';')
    transactions_df = pd.read_csv(os.path.join(base_dir, "transaction.csv"), sep=';')
    transaction_items_df = pd.read_csv(os.path.join(base_dir, "transaction_items.csv"), sep=';')
    
    # Extract product categories from product names
    products_df['category'] = products_df['category'].astype(str)
    # Extract main category (simplified for synthetic data)
    def extract_category(category_str):
        # Simple categorization based on the first few words
        lower_cat = category_str.lower()
        if 'baju pantai' in lower_cat:
            return 1
        elif 'celana pantai' in lower_cat:
            return 2
        elif 'baju kemeja' in lower_cat:
            return 3
        elif 'gelang' in lower_cat:
            return 4
        elif 'kalung' in lower_cat:
            return 5
        elif 'kamen' in lower_cat:
            return 6
        else:
            return 7
    
    # Update product categories to numeric values
    products_df['category_id'] = products_df['category'].apply(extract_category)
    
    # Define parameters for generating data
    start_date = datetime(2019, 1, 1)  # Start from January 2024
    end_date = datetime(2024, 12, 31)  # Until end of 2024
    
    # Generate more transaction data
    new_transactions = []
    new_transaction_items = []
    
    # Get the highest existing transaction_id
    max_transaction_id = transactions_df['transaction_id'].max() + 1
    
    # Get the highest existing product_id
    max_product_id = products_df['product_id'].max()

    # Customer IDs
    customer_ids = customers_df['customer_id'].tolist()
    
    # Payment methods
    payment_methods = ['e-wallet', 'credit_card', 'cash']
    
    # Generate transactions to reach target size
    num_to_generate = max(0, target_size - len(transactions_df))
    print(f"Generating {num_to_generate} new transactions...")
    
    for i in range(num_to_generate):
        # Generate transaction
        transaction_id = max_transaction_id + i
        customer_id = random.choice(customer_ids)
        
        # Evenly distribute across years 2019-2024
        year = 2019 + (i % 6)  # This will cycle through years 2019, 2020, 2021, 2022, 2023, 2024
        
        # Generate date in the selected year, ensuring proper distribution across quarters
        quarter = random.randint(1, 4)
        month = ((quarter - 1) * 3) + random.randint(1, 3)  # Map quarter to corresponding months
        day = random.randint(1, 28)  # Avoid invalid dates
        date = f"{day:02d}/{month:02d}/{year}"  # Use the selected year instead of hardcoding 2024
        
        # Generate a realistic transaction total (will be updated based on items)
        base_total = random.uniform(50000, 2000000)
        payment_method = random.choice(payment_methods)
        
        # Add transaction
        new_transactions.append({
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'date': date,
            'total_amount': base_total,
            'payment_method': payment_method
        })
        
        # Generate 1-3 items for this transaction
        num_items = random.randint(1, 3)
        transaction_total = 0
        
        for j in range(num_items):
            # Choose a product
            product_id = random.randint(1, max_product_id)
            
            # Set quantity and price
            quantity = random.randint(1, 3)
            
            # Get actual product price if available, otherwise generate random price
            product_price = products_df.loc[products_df['product_id'] == product_id, 'price'].values
            if len(product_price) > 0:
                unit_price = product_price[0]
            else:
                unit_price = random.uniform(100000, 1000000)
            
            transaction_total += quantity * unit_price
            
            # Add transaction item
            new_transaction_items.append({
                'transaction_id': transaction_id,
                'product_id': product_id,
                'quantity': quantity,
                'unit_price': unit_price
            })
        
        # Update transaction total amount based on actual items
        new_transactions[-1]['total_amount'] = transaction_total
    
    # Create new DataFrames
    new_transactions_df = pd.DataFrame(new_transactions)
    new_transaction_items_df = pd.DataFrame(new_transaction_items)
    
    # Combine with existing data
    transactions_df = pd.concat([transactions_df, new_transactions_df], ignore_index=True)
    transaction_items_df = pd.concat([transaction_items_df, new_transaction_items_df], ignore_index=True)
    
    # Standardize data types
    transactions_df['transaction_id'] = transactions_df['transaction_id'].astype(int)
    transaction_items_df['transaction_id'] = transaction_items_df['transaction_id'].astype(int)
    transaction_items_df['product_id'] = transaction_items_df['product_id'].astype(int)
    
    print(f"Generated dataset contains {len(transactions_df)} transactions with {len(transaction_items_df)} line items")
    
    # Save the extended data to CSV files
    transactions_df.to_csv(os.path.join(base_dir, "transaction_extended.csv"), sep=';', index=False)
    transaction_items_df.to_csv(os.path.join(base_dir, "transaction_items_extended.csv"), sep=';', index=False)
    
    # Generate preference scores for XGBoost training
    print("Generating preference scores for transactions...")
    preference_data = []
    
    # Join transaction data with customer and product info
    for _, item in transaction_items_df.iterrows():
        transaction_id = item['transaction_id']
        product_id = item['product_id']
        
        # Get transaction info
        transaction = transactions_df[transactions_df['transaction_id'] == transaction_id]
        if transaction.empty:
            continue
            
        transaction = transaction.iloc[0]
        customer_id = transaction['customer_id']
        date_str = transaction['date']
        
        # Parse date
        try:
            date = datetime.strptime(date_str, '%d/%m/%Y')
            year = date.year
            quarter = ((date.month - 1) // 3) + 1
        except:
            # If date parsing fails, assign random quarter
            year = random.randint(2019, 2024)  # Fix typo: randiant -> randint
            quarter = random.randint(1, 4)
        
        # Get customer info
        customer = customers_df[customers_df['customer_id'] == customer_id]
        if customer.empty:
            continue
        customer = customer.iloc[0]
        
        # Get product info
        product = products_df[products_df['product_id'] == product_id]
        if product.empty:
            continue
        product = product.iloc[0]
        
        # Generate a preference score based on customer and product attributes
        # Higher membership levels tend to have higher preference scores
        membership_bonus = {'bronze': 0, 'silver': 0.1, 'gold': 0.2, 'platinum': 0.3}.get(
            customer['membership_level'].lower(), 0)
        
        # Calculate base preference (a number between 0 and 1)
        base_preference = random.uniform(0.2, 0.8)
        
        # Add some randomness based on product category
        category_preference = random.uniform(-0.1, 0.1)
        
        # Combine factors for final preference score
        preference_score = min(1.0, max(0.0, base_preference + membership_bonus + category_preference))
        
        # Add a small seasonal effect by quarter
        seasonal_effect = {
            1: random.uniform(-0.05, 0.05),  # Q1
            2: random.uniform(-0.05, 0.05),  # Q2
            3: random.uniform(-0.05, 0.05),  # Q3
            4: random.uniform(-0.05, 0.05)   # Q4
        }[quarter]
        
        preference_score = min(1.0, max(0.0, preference_score + seasonal_effect))
        
        # Create a preference record
        preference_data.append({
            'customer_id': customer_id,
            'product_id': product_id,
            'year': year,
            'quarter': quarter,
            'preference_score': preference_score,
            'age': customer['age'],
            'gender': customer['gender'],
            'city': customer['city'],
            'membership_level': customer['membership_level'],            
            'category': product.get('category_id', 0),  # Use numeric category if available
            'brand': product['brand'],
            'price': product['price'],
            'quantity': item['quantity'],
            'unit_price': item['unit_price'],
            'total_amount': transaction['total_amount']
        })
    
    # Create preference DataFrame
    preference_df = pd.DataFrame(preference_data)
    print(f"Generated {len(preference_df)} preference records")
    
    # Save preference data
    preference_df.to_csv(os.path.join(base_dir, "preference_data.csv"), index=False)
    
    return transactions_df, transaction_items_df, preference_df

# Modify xgboost_model.py to use synthetic data
def modify_xgboost_model():
    """
    Add code to the xgboost_model.py file to use the synthetic data
    """
    xgboost_model_path = "xgboost_model.py"
    
    # Read the existing file
    with open(xgboost_model_path, 'r') as file:
        content = file.read()
    
    # Check if the file already includes synthetic data handling
    if "preference_data.csv" in content:
        print("xgboost_model.py already modified to use synthetic data")
        return
    
    # Add synthetic data loading to the load_data function
    modified_content = content.replace(
        "def load_data():",
        """def load_data():
    # Check if synthetic data is available
    if os.path.exists('data/preference_data.csv'):
        print("Loading synthetic preference data...")
        preference_df = pd.read_csv('data/preference_data.csv')
        return preference_df
    
    # If no synthetic data, proceed with original data loading
"""
    )
    
    # Write the modified file
    with open(xgboost_model_path, 'w') as file:
        file.write(modified_content)
    
    print("Modified xgboost_model.py to use synthetic data when available")

if __name__ == "__main__":
    print("Generating synthetic data for The X Aksha Buyer Preference Analysis...")
    transactions_df, transaction_items_df, preference_df = generate_synthetic_data(target_size=1000)
    modify_xgboost_model()
    print("Data generation complete. You can now run xgboost_model.py followed by visualisasi.py")