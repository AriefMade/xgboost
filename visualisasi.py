import pandas as pd
import numpy as np
import dash
from dash import dcc, html, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
import pickle
import os
import sys
sys.path.append('d:\\Proposal & Tugas akhir\\aplikasi\\xgboost')
import xgboost_model

try:
    # Try to load pickled model and data if they exist
    if os.path.exists('quarterly_models.pkl') and os.path.exists('processed_data.pkl') and os.path.exists('recommendations.pkl'):
        print("Loading existing model and processed data...")
        with open('quarterly_models.pkl', 'rb') as f:
            quarterly_models = pickle.load(f)
        df = pd.read_pickle('processed_data.pkl')
        recommendations = pd.read_pickle('recommendations.pkl')
    else:
        # Run the main function to generate the model and data
        print("Processing data and training models...")
        quarterly_models, df, recommendations = xgboost_model.main()
except Exception as e:
    print(f"Error initializing: {e}")
    sys.exit(1)

# Print key information about the loaded models for debugging
print(f"Loaded {len(quarterly_models)} quarterly models")
print(f"Model keys: {list(quarterly_models.keys())}")
print(f"Key types: {[type(k).__name__ for k in quarterly_models.keys()]}")

# Buat mapping kategori berdasarkan kategori unik dalam data
# Jika menggunakan data CSV, ambil kategori unik
unique_categories = df['category'].unique()
category_mapping = {i: f"Category {i}" for i in unique_categories}

# Inisialisasi aplikasi Dash
app = dash.Dash(__name__, title='The X Aksha Buyer Preference Dashboard')

# Definisikan layout dashboard
app.layout = html.Div([
    html.H1('The X Aksha - Analisis Preferensi Pembeli', style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3('Pilih Parameter Analisis'),
            html.Label('Tahun:'),
            dcc.Dropdown(
                id='year-dropdown',
                options=[
                    {'label': '2019', 'value': 2019},
                    {'label': '2020', 'value': 2020},
                    {'label': '2021', 'value': 2021},
                    {'label': '2022', 'value': 2022},
                    {'label': '2023', 'value': 2023},
                    {'label': '2024', 'value': 2024}
                ],
                value=2023,
                clearable=False 
            ),
            html.Label('Kuartal:'),
            dcc.Dropdown(
                id='quarter-dropdown',
                options=[
                    {'label': 'Q1', 'value': 1},
                    {'label': 'Q2', 'value': 2},
                    {'label': 'Q3', 'value': 3},
                    {'label': 'Q4', 'value': 4}
                ],
                value=1,
                clearable=False
            ),
            
            html.Label('Kategori Produk:'),
            dcc.Dropdown(
                id='category-dropdown',
                options=[
                    {'label': 'Semua Kategori', 'value': 'all'},
                    *[{'label': cat_name, 'value': cat_id} for cat_id, cat_name in category_mapping.items()]
                ],
                value='all',
                clearable=False
            ),
            
            html.Label('Level Membership:'),
            dcc.Dropdown(
                id='membership-dropdown',
                options=[
                    {'label': 'Semua Level', 'value': 'all'},
                    {'label': 'Bronze', 'value': 0},
                    {'label': 'Silver', 'value': 1},
                    {'label': 'Gold', 'value': 2},
                    {'label': 'Platinum', 'value': 3}
                ],
                value='all',
                clearable=False
            ),
            
            html.Div([
                html.Label('Membership Level untuk Rekomendasi Penjualan:'),
                dcc.Dropdown(
                    id='membership-recommendation-dropdown',
                    options=[
                        {'label': 'Bronze', 'value': 0},
                        {'label': 'Silver', 'value': 1},
                        {'label': 'Gold', 'value': 2},
                        {'label': 'Platinum', 'value': 3}
                    ],
                    value=0,
                    clearable=False
                ),
                html.Button('Analisis Tren Penjualan', id='submit-button', n_clicks=0)
            ], style={'marginTop': '20px', 'marginBottom': '20px'})
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '20px'}),
        
        html.Div([
            html.H3('Insight Preferensi Pembeli'),
            dcc.Tabs([
                dcc.Tab(label='Overview', children=[
                    dcc.Graph(id='preference-overview')
                ]),
                dcc.Tab(label='Tren Kuartalan', children=[
                    dcc.Graph(id='quarterly-trends')
                ]),
                dcc.Tab(label='Feature Importance', children=[
                    dcc.Graph(id='feature-importance')
                ]),
                dcc.Tab(label='Perbandingan Tahunan', children=[
                    dcc.Graph(id='yearly-comparison')
                ])
            ])
        ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
    ]),
    
    html.Div([
        html.H3('Rekomendasi Produk Berdasarkan Preferensi', style={'textAlign': 'center'}),
        html.Div(id='recommendations-output')
    ], style={'padding': '20px'}),
    
    html.Div([
        html.H3('Rekomendasi Bisnis', style={'textAlign': 'center'}),
        html.Div(id='business-recommendations')
    ], style={'padding': '20px'})
])

# Callback untuk update visualisasi overview
@callback(
    Output('preference-overview', 'figure'),
    [Input('year-dropdown','value'),
     Input('quarter-dropdown', 'value'),
     Input('category-dropdown', 'value'),
     Input('membership-dropdown', 'value')]
)
def update_overview(year, quarter, category, membership):
    try:
        # Filter data berdasarkan input
        filtered_df = df[(df['year']== year) & (df['quarter'] == quarter)]
        
        if category != 'all':
            filtered_df = filtered_df[filtered_df['category'] == category]
        
        if membership != 'all':
            filtered_df = filtered_df[filtered_df['membership_level'] == membership]
        
        # Handle kasus ketika tidak ada data
        if filtered_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Tidak ada data yang tersedia untuk filter yang dipilih",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
            
        # Buat subplot dengan 2 baris, 2 kolom
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribusi Skor Preferensi', 
                'Preferensi berdasarkan Kategori',
                'Preferensi berdasarkan Membership Level', 
                'Preferensi berdasarkan Usia'
            )
        )
        
        # 1. Distribusi skor preferensi
        fig.add_trace(
            go.Histogram(
                x=filtered_df['preference_score'],
                name='Distribusi Preferensi',
                marker_color='royalblue'
            ),
            row=1, col=1
        )
        
        # 2. Preferensi berdasarkan kategori
        cat_pref = filtered_df.groupby('category')['preference_score'].mean().reset_index()
        cat_pref['category_name'] = cat_pref['category'].map(category_mapping)  # Tambahkan pemetaan nama kategori
        
        fig.add_trace(
            go.Bar(
                x=cat_pref['category_name'],  # Ubah ke category_name
                y=cat_pref['preference_score'],
                name='Preferensi per Kategori',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # 3. Preferensi berdasarkan membership level
        mem_pref = filtered_df.groupby('membership_level')['preference_score'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=mem_pref['membership_level'],
                y=mem_pref['preference_score'],
                name='Preferensi per Membership',
                marker_color='coral'
            ),
            row=2, col=1
        )
        
        # 4. Preferensi berdasarkan usia
        # Buat bin usia
        filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
        age_pref = filtered_df.groupby('age_group')['preference_score'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=age_pref['age_group'],
                y=age_pref['preference_score'],
                name='Preferensi per Usia',
                marker_color='mediumpurple'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text=f"Overview Preferensi Pembeli - Q{quarter} {year}"  # Add year here
        )
        
        return fig
    except Exception as e:
        # Handle exception
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=f"Error: {str(e)}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return empty_fig

# Callback untuk update tren kuartalan
@callback(
    Output('quarterly-trends', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('year-dropdown', 'value')]  # Add year as input
)
def update_quarterly_trends(category, year):  # Add year parameter
    # Siapkan data tren kuartalan
    # Filter by year as well
    if category == 'all':
        category_filter = df[df['year'] == year]
    else:
        category_filter = df[(df['category'] == category) & (df['year'] == year)]
    
    # If there's no data for the selected year, try to use data from any year
    if category_filter.empty:
        if category == 'all':
            category_filter = df
        else:
            category_filter = df[df['category'] == category]
        
        # Still empty? Return an empty figure
        if category_filter.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title=f"No data available for {category} in {year}",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
    
    quarterly_trends = category_filter.groupby(['quarter', 'category'])['preference_score'].mean().reset_index()
    
    # Plot tren
    fig = go.Figure()
    
    for cat in quarterly_trends['category'].unique():
        cat_data = quarterly_trends[quarterly_trends['category'] == cat]
        fig.add_trace(
            go.Scatter(
                x=cat_data['quarter'],
                y=cat_data['preference_score'],
                mode='lines+markers',
                name=f'Kategori {cat}'
            )
        )
    
    # Tambahkan perubahan persentase antar kuartal
    if category != 'all':
        # Jika hanya satu kategori dipilih, tambahkan anotasi perubahan
        cat_data = quarterly_trends[quarterly_trends['category'] == category]
        for i in range(len(cat_data)-1):
            if i+1 < len(cat_data):  # Make sure next index exists
                q_current = cat_data.iloc[i]['quarter']
                q_next = cat_data.iloc[i+1]['quarter']
                score_current = cat_data.iloc[i]['preference_score']
                score_next = cat_data.iloc[i+1]['preference_score']
                pct_change = (score_next - score_current) / score_current * 100
                
                fig.add_annotation(
                    x=(q_current + q_next) / 2,
                    y=(score_current + score_next) / 2,
                    text=f"{pct_change:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363"
                )
    
    fig.update_layout(
        title=f"Tren Preferensi Pembeli per Kuartal {year} - {'Semua Kategori' if category == 'all' else f'Kategori {category}'}",
        xaxis_title="Kuartal",
        yaxis_title="Skor Preferensi Rata-rata",
        legend_title="Kategori",
        height=500
    )
    
    # Use dynamic year in labels
    fig.update_xaxes(
        tickvals=[1, 2, 3, 4],
        ticktext=[f"Q1 {year}", f"Q2 {year}", f"Q3 {year}", f"Q4 {year}"]
    )
    
    return fig

# Callback untuk feature importance
@callback(
    Output('feature-importance', 'figure'),
    [Input('year-dropdown', 'value'),  # Add year as input
     Input('quarter-dropdown', 'value')]
)
def update_feature_importance(year, quarter):
    # Find available quarters based on format in quarterly_models
    available_quarters = list(quarterly_models.keys())
    
    if not available_quarters:
        return go.Figure().update_layout(
            title="Tidak ada model yang tersedia",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    
    # Check if keys are in format like "2024Q1" or just integers
    if all(isinstance(k, str) and "Q" in k for k in available_quarters):
        # Keys are in format like "2024Q1"
        quarter_key = f"{year}Q{quarter}"
    else:
        # Keys are simple integers or other format
        quarter_key = quarter
    
    # Check if the specific quarter model exists
    if quarter_key not in quarterly_models:
        # Find the closest available quarter
        print(f"Model for {quarter_key} not found. Available models: {available_quarters}")
        
        # For string keys like "2024Q1"
        if all(isinstance(k, str) and "Q" in k for k in available_quarters):
            # Filter by year first
            year_quarters = [q for q in available_quarters if q.startswith(f"{year}Q")]
            if year_quarters:
                quarter_key = year_quarters[0]  # Use first quarter from same year
            else:
                quarter_key = available_quarters[0]  # Default to first available
        else:
            # For numeric quarter keys
            quarter_key = available_quarters[0]  # Default to first available
    
    # Get the model for the selected quarter
    model = quarterly_models[quarter_key]
    
    try:
        # Get feature importance
        importance = model.get_score(importance_type='gain')
        features = list(importance.keys())
        scores = list(importance.values())
        
        # Create DataFrame and sort by score
        importance_df = pd.DataFrame({'Feature': features, 'Importance': scores})
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        if len(importance_df) > 0:
            # Show all features if less than 10
            features_to_show = min(10, len(importance_df))
            
            # Plot horizontal bar chart
            fig = go.Figure(go.Bar(
                y=importance_df['Feature'].tail(features_to_show),
                x=importance_df['Importance'].tail(features_to_show),
                orientation='h',
                marker=dict(
                    color='rgba(58, 71, 80, 0.8)',
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=2)
                )
            ))
            
            # Extract quarter number for display
            if isinstance(quarter_key, str) and "Q" in quarter_key:
                display_quarter = quarter_key.split("Q")[1]
                display_year = quarter_key.split("Q")[0]
                title = f'Top {features_to_show} Feature Importance - {display_year} Q{display_quarter}'
            else:
                title = f'Top {features_to_show} Feature Importance - Q{quarter} {year}'
            
            fig.update_layout(
                title_text=title,
                xaxis_title='Importance Score (Gain)',
                yaxis_title='Feature',
                height=500
            )
            
            return fig
        else:
            raise ValueError("No features found in model")
            
    except Exception as e:
        # Return empty figure with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Tidak dapat menampilkan feature importance: {str(e)}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

@callback(
    Output('yearly-comparison', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('membership-dropdown', 'value')]
)
def update_yearly_comparison(category, membership):
    try:
        # Filter data berdasarkan input
        if category != 'all':
            filtered_df = df[df['category'] == category]
        else:
            filtered_df = df.copy()
            
        if membership != 'all':
            filtered_df = filtered_df[filtered_df['membership_level'] == membership]
        
        # Handle kasus ketika tidak ada data
        if filtered_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Tidak ada data yang tersedia untuk filter yang dipilih",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return empty_fig
        
        # Menghitung rata-rata preferensi per tahun dan kuartal
        yearly_comparison = filtered_df.groupby(['year', 'quarter'])['preference_score'].mean().reset_index()
        
        # Tambahkan kolom untuk label
        yearly_comparison['period'] = yearly_comparison.apply(lambda x: f"Q{x['quarter']} {x['year']}", axis=1)
        
        # Urutkan berdasarkan tahun dan kuartal
        yearly_comparison = yearly_comparison.sort_values(['year', 'quarter'])
        
        # Buat figure
        fig = go.Figure()
        
        # Plot comparison bar chart
        fig.add_trace(
            go.Bar(
                x=yearly_comparison['period'],
                y=yearly_comparison['preference_score'],
                name='Skor Preferensi',
                marker_color='royalblue',
                text=yearly_comparison['preference_score'].round(2),
                textposition='auto'
            )
        )
        
        # Tambah garis tren untuk visualisasi yang lebih baik
        fig.add_trace(
            go.Scatter(
                x=yearly_comparison['period'],
                y=yearly_comparison['preference_score'],
                mode='lines+markers',
                name='Tren',
                line=dict(color='red', width=2)
            )
        )
        
        # Menambahkan anotasi perubahan persentase antar periode yang berurutan
        for i in range(len(yearly_comparison) - 1):
            current = yearly_comparison.iloc[i]
            next_period = yearly_comparison.iloc[i+1]
            
            # Hitung perubahan persentase
            pct_change = ((next_period['preference_score'] - current['preference_score']) / 
                          current['preference_score'] * 100)
            
            # Tambahkan anotasi hanya jika perubahan signifikan (>1%)
            if abs(pct_change) > 1:
                fig.add_annotation(
                    x=(i + i+1) / 2,  # Posisi x di tengah antara dua periode
                    y=max(current['preference_score'], next_period['preference_score']) + 0.05,
                    text=f"{pct_change:.1f}%" if pct_change != 0 else "0%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="gray",
                    arrowsize=1,
                    arrowwidth=1,
                    ax=0,
                    ay=-30,
                    font=dict(size=10)
                )
        
        # Update layout
        title_text = f"Perbandingan Skor Preferensi Tahunan - "
        if category != 'all':
            cat_name = category_mapping.get(category, f"Kategori {category}")
            title_text += cat_name
        else:
            title_text += "Semua Kategori"
            
        if membership != 'all':
            membership_names = {0: 'Bronze', 1: 'Silver', 2: 'Gold', 3: 'Platinum'}
            title_text += f" - {membership_names.get(membership, f'Level {membership}')}"
        
        fig.update_layout(
            title=title_text,
            xaxis_title="Periode (Kuartal-Tahun)",
            yaxis_title="Skor Preferensi Rata-rata",
            legend_title="Legenda",
            height=500,
            barmode='group',
            # Tambahkan custom range pada sumbu y untuk visualisasi yang lebih baik
            yaxis=dict(
                range=[
                    max(0, yearly_comparison['preference_score'].min() * 0.9),  # Mulai dari 0 atau 90% nilai minimum
                    yearly_comparison['preference_score'].max() * 1.1  # Hingga 110% nilai maksimum
                ]
            )
        )
        
        return fig
        
    except Exception as e:
        # Tangani exception
        print(f"Error in yearly comparison: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=f"Error: {str(e)}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return empty_fig

# Callback untuk rekomendasi produk
@callback(
    Output('recommendations-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('membership-recommendation-dropdown', 'value'),
     dash.dependencies.State('year-dropdown', 'value'),
     dash.dependencies.State('quarter-dropdown', 'value')]
)
def update_recommendations(n_clicks, membership_level, year, quarter):
    if n_clicks == 0:
        return html.Div("Pilih Level Membership dan klik 'Analisis Tren Penjualan' untuk melihat rekomendasi")
    
    try:
        # Tampilkan nama membership
        membership_names = {0: 'Bronze', 1: 'Silver', 2: 'Gold', 3: 'Platinum'}
        membership_name = membership_names.get(membership_level, f"Level {membership_level}")
        
        # Dapatkan rekomendasi berdasarkan tren historis
        recommendations = get_quarterly_product_recommendations(
            membership_level, quarter, df, category_mapping
        )
        
        if not recommendations:
            return html.Div(f"Tidak cukup data historis untuk membership {membership_name} pada Q{quarter}")
        
        # Buat tampilan rekomendasi yang lebih informatif
        recommendation_cards = []
        
        # Header untuk hasil analisis
        recommendation_cards.append(html.Div([
            html.H4(f"Analisis Tren Penjualan - Membership {membership_name} - Q{quarter} (Historis)", 
                   style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px', 'borderRadius': '5px'}),
            html.P([
                f"Menampilkan analisis berdasarkan data historis kuartal {quarter} dari berbagai tahun. ",
                html.Br(),
                f"Gunakan rekomendasi ini untuk menyiapkan inventaris Anda untuk Q{quarter+1 if quarter < 4 else 1}."
            ])
        ]))
        
        # Kartu kategori
        for i, rec in enumerate(recommendations):
            trend_color = "green" if rec['trend_direction'] == "Naik" else \
                          "red" if rec['trend_direction'] == "Turun" else "gray"
            
            # Buat tabel produk rekomendasi
            product_rows = []
            for prod in rec['recommended_products']:
                product_rows.append(html.Tr([
                    html.Td(f"{prod['product_id']}"),
                    html.Td(f"{prod['brand']}"),
                    html.Td(f"Rp {prod['price']:,.0f}"),
                    html.Td(f"{prod['preference_score']:.2f}")
                ]))
            
            product_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("ID Produk"),
                    html.Th("Brand"),
                    html.Th("Harga"),
                    html.Th("Skor Pref.")
                ])),
                html.Tbody(product_rows)
            ], style={
                'width': '100%',
                'border': '1px solid #ddd',
                'borderCollapse': 'collapse',
                'marginTop': '10px'
            })
            
            # Warna latar belakang bergantian untuk kartu
            bg_color = '#f9f9f9' if i % 2 == 0 else 'white'
            
            card = html.Div([
                html.Div([
                    html.H5(rec['category_name'], style={'margin': '0'}),
                    html.Span(f"{rec['trend_direction']} ({rec['trend_value']:.1f}%)", 
                              style={'color': trend_color, 'fontWeight': 'bold', 'float': 'right'})
                ], style={'borderBottom': '1px solid #ddd', 'paddingBottom': '8px'}),
                
                html.Div([
                    html.P([
                        f"Preferensi rata-rata: ",
                        html.B(f"{rec['avg_preference']:.2f}")
                    ]),
                    html.P([
                        f"Jumlah transaksi: ",
                        html.B(f"{rec['transaction_count']}")
                    ]),
                    html.P([
                        html.B("Produk yang direkomendasikan untuk Q{quarter+1 if quarter < 4 else 1}:")
                    ]),
                    product_table
                ])
            ], style={
                'border': '1px solid #ddd',
                'borderRadius': '4px',
                'padding': '15px',
                'marginBottom': '15px',
                'backgroundColor': bg_color,
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
            
            recommendation_cards.append(card)
        
        # Tambahkan rangkuman strategis
        top_category = recommendations[0]['category_name'] if recommendations else "tidak diketahui"
        
        strategy_card = html.Div([
            html.H4("Ringkasan Strategis", style={'backgroundColor': '#2196F3', 'color': 'white', 'padding': '10px', 'borderRadius': '5px'}),
            html.Div([
                html.P([
                    html.B("Fokus utama: "), 
                    f"Perkuat inventaris kategori {top_category} untuk pelanggan {membership_name}"
                ]),
                html.P([
                    html.B("Rekomendasi stok: "), 
                    f"Untuk Q{quarter+1 if quarter < 4 else 1}, tingkatkan persediaan produk dari brand-brand terpopuler dalam kategori teratas"
                ]),
                html.P([
                    html.B("Strategi pemasaran: "), 
                    f"Buat kampanye yang ditargetkan untuk pelanggan {membership_name} dengan fokus pada kategori teratas"
                ]),
            ], style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '4px', 'backgroundColor': 'white'})
        ], style={'marginTop': '20px', 'marginBottom': '20px'})
        
        recommendation_cards.append(strategy_card)
        
        return html.Div(recommendation_cards)
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return html.Div([
            html.H4("Error menganalisis tren penjualan", style={'color': 'red'}),
            html.Pre(error_msg, style={'whiteSpace': 'pre-wrap', 'backgroundColor': '#f8f9fa', 'padding': '10px'})
        ])

# Callback untuk rekomendasi bisnis
@callback(
    Output('business-recommendations', 'children'),
    [Input('quarter-dropdown', 'value'),
     Input('year-dropdown', 'value')]  # Add year input
)
def update_business_recommendations(quarter, year):
    # Check if recommendations exist
    if recommendations is None or recommendations.empty:
        return html.Div("Tidak ada rekomendasi bisnis tersedia.")
    
    try:
        # Extract available years and quarters from recommendations to check validity
        available_periods = recommendations['period'].unique()
        year_quarters = []
        for period in available_periods:
            if "Q" in period and " " in period:
                q_part = period.split(" ")[0].replace("Q", "")
                y_part = period.split(" ")[1]
                if q_part.isdigit() and y_part.isdigit():
                    year_quarters.append((int(y_part), int(q_part)))
        
        # Check if selected year-quarter is available
        has_exact_period = any(y == year and q == quarter for y, q in year_quarters)
        has_year = any(y == year for y, q in year_quarters)
        
        # Filter recommendations based on available data
        if has_exact_period:
            # Filter for exact year and quarter
            period_str = f"Q{quarter} {year}"
            quarter_recommendations = recommendations[recommendations['period'] == period_str]
        elif has_year:
            # Only matching year is available
            quarter_recommendations = recommendations[recommendations['period'].str.contains(f"{year}")]
        else:
            # Neither year nor quarter matches, show a message
            quarter_recommendations = pd.DataFrame()
        
        # If no recommendations for this quarter, show for any period
        if quarter_recommendations.empty:
            return html.Div([
                html.P(f"Tidak ada rekomendasi bisnis untuk Tahun {year}, Kuartal {quarter}. Menampilkan rekomendasi umum:"),
                html.Div([
                    html.H4("Rekomendasi Umum", style={'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px', 'borderRadius': '5px 5px 0 0'}),
                    html.Div([
                        html.H5("Catatan", style={'fontWeight': 'bold'}),
                        html.P(f"Tidak tersedia data spesifik untuk tahun {year}, kuartal {quarter}. Gunakan data tahun dan kuartal lain atau tambahkan lebih banyak data untuk hasil yang lebih baik.")
                    ], style={'padding': '15px', 'border': '1px solid #ddd', 'borderTop': 'none', 'borderRadius': '0 0 5px 5px'})
                ])
            ])
        
        # Kelompokkan berdasarkan tipe rekomendasi
        recommendation_types = quarter_recommendations['type'].unique()
        
        recommendation_cards = []
        for rec_type in recommendation_types:
            type_recs = quarter_recommendations[quarter_recommendations['type'] == rec_type]
            
            rec_items = []
            for _, rec in type_recs.iterrows():
                rec_items.append(html.Div([
                    html.H5(rec['finding'], style={'fontWeight': 'bold'}),
                    html.P(rec['recommendation'])
                ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}))
            
            # Buat card untuk tipe rekomendasi (add period to header)
            recommendation_cards.append(html.Div([
                html.H4(f"Rekomendasi {rec_type} - {quarter_recommendations['period'].iloc[0]}", 
                       style={'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px', 'borderRadius': '5px 5px 0 0'}),
                html.Div(rec_items, style={'padding': '15px', 'border': '1px solid #ddd', 'borderTop': 'none', 'borderRadius': '0 0 5px 5px'})
            ], style={'marginBottom': '20px'}))
        
        return html.Div(recommendation_cards)
    
    except Exception as e:
        return html.Div(f"Error menampilkan rekomendasi: {str(e)}")

# Fungsi untuk prediksi preferensi customer (sama dengan di file sebelumnya)
def predict_customer_preferences(customer_id, quarter, models, df, year=None, top_n=5):
    """
    Memprediksi preferensi produk untuk pelanggan tertentu pada kuartal dan tahun tertentu.
    """
    # Check if any models are available
    available_quarters = list(models.keys())
    if not available_quarters:
        print("No models available.")
        return None
    
    # Print model key information for debugging
    print(f"Available quarters: {available_quarters}")
    print(f"Types: {[type(k).__name__ for k in available_quarters]}")
    
    # Determine if we're working with string keys like "2024Q1" or integer keys
    string_keys = any(isinstance(k, str) and "Q" in k for k in available_quarters)
    
    # Get current year from data if not provided
    if year is None:
        year = df['year'].max() if not df.empty and 'year' in df.columns else 2024
    
    # Convert quarter to the appropriate format based on model keys
    if string_keys:
        quarter_key = f"{year}Q{quarter}"
        print(f"Looking for model with key: {quarter_key}")
    else:
        quarter_key = quarter
        print(f"Looking for model with key: {quarter_key}")
    
    # If requested quarter isn't available, find the best alternative
    if quarter_key not in models:
        print(f"Model for {quarter_key} not available.")
        closest_quarter = None
        
        if string_keys:
            # First try to find a model from the same year
            same_year_keys = [k for k in available_quarters if isinstance(k, str) and k.startswith(f"{year}Q")]
            if same_year_keys:
                closest_quarter = same_year_keys[0]
                print(f"Using model from same year: {closest_quarter}")
            else:
                # Try to find the closest year with the same quarter
                year_quarter_pairs = []
                for k in available_quarters:
                    if isinstance(k, str) and "Q" in k:
                        try:
                            y, q = k.split("Q")
                            if q.isdigit() and y.isdigit():
                                year_quarter_pairs.append((int(y), int(q), k))
                        except ValueError:
                            continue
                
                # Sort by year difference, then quarter difference
                if year_quarter_pairs:
                    closest = min(year_quarter_pairs, 
                                 key=lambda yqk: (abs(yqk[0] - year), abs(yqk[1] - quarter)))
                    closest_quarter = closest[2]
                    print(f"Using closest available model: {closest_quarter}")
                else:
                    closest_quarter = available_quarters[0]
                    print(f"No similar models found. Using first available: {closest_quarter}")
        else:
            # For numeric keys, find the numerically closest quarter
            try:
                closest_quarter = min(available_quarters, 
                                     key=lambda x: abs(int(x) - int(quarter)) if str(x).isdigit() else float('inf'))
                print(f"Using closest quarter: {closest_quarter}")
            except (ValueError, TypeError):
                closest_quarter = available_quarters[0]
                print(f"Error finding closest quarter. Using first available: {closest_quarter}")
        
        quarter_key = closest_quarter
    
    # Get model for the quarter
    model = models[quarter_key]
    
    # Extract year and quarter for filtering data
    if isinstance(quarter_key, str) and "Q" in quarter_key:
        try:
            filter_year = int(quarter_key.split("Q")[0])
            filter_quarter = int(quarter_key.split("Q")[1])
        except (ValueError, IndexError):
            filter_year = year
            filter_quarter = quarter
    else:
        filter_year = year
        filter_quarter = quarter if isinstance(quarter, int) else 1
    
    print(f"Filtering data for year {filter_year}, quarter {filter_quarter}")
    
    # Try to find customer data for the specific year/quarter
    customer_data = df[(df['customer_id'] == customer_id) & 
                     (df['year'] == filter_year) & 
                     (df['quarter'] == filter_quarter)]
    
    # If no data for that customer in that year/quarter, try other quarters from same year
    if customer_data.empty:
        print(f"No data for customer {customer_id} in Q{filter_quarter} {filter_year}")
        customer_data = df[(df['customer_id'] == customer_id) & (df['year'] == filter_year)]
        
        # If still no data, try any year
        if customer_data.empty:
            print(f"No data for customer {customer_id} in {filter_year}. Trying any year...")
            customer_data = df[df['customer_id'] == customer_id]
            
            # If still no data, try any customer in that year/quarter
            if customer_data.empty:
                print(f"No data for customer {customer_id}. Using other customers from {filter_year} Q{filter_quarter}...")
                customer_data = df[(df['year'] == filter_year) & (df['quarter'] == filter_quarter)].head(top_n)
                
                # Last resort: use any available data
                if customer_data.empty:
                    print("No suitable data found. Using any available data...")
                    customer_data = df.head(top_n)
    
    if customer_data.empty:
        print("No data available at all.")
        return None
    
    print(f"Using {len(customer_data)} transaction records for prediction")
    
    # Required features
    features = [
        'age', 'gender', 'city', 'membership_level', 'category', 'brand', 'price',
        'day_of_week', 'is_weekend', 'month', 'price_per_quantity', 'price_ratio', 
        'total_amount'
    ]
    
    # Check for missing features
    missing_features = [f for f in features if f not in customer_data.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        return None
    
    # Predict preferences
    results = []
    for _, row in customer_data.iterrows():
        try:
            X = row[features].values.reshape(1, -1)
            dtest = xgb.DMatrix(X)
            pred_score = model.predict(dtest)[0]
            
            results.append({
                'product_id': row['product_id'],
                'category': row['category'],
                'brand': row['brand'],
                'predicted_preference': pred_score,
                'actual_preference': row['preference_score']
            })
        except Exception as e:
            print(f"Prediction error for product {row.get('product_id', 'unknown')}: {e}")
    
    # Sort by predicted preference
    if not results:
        return None
        
    results_df = pd.DataFrame(results)
    top_preferences = results_df.sort_values('predicted_preference', ascending=False).head(top_n)
    
    return top_preferences

def analyze_membership_sales_trends(membership_level, quarter, df):
    """
    Menganalisis tren penjualan historis untuk level membership tertentu pada kuartal tertentu
    di beberapa tahun berbeda.
    """
    try:
        # Filter data untuk membership level yang dipilih pada kuartal tertentu dari semua tahun
        membership_data = df[(df['membership_level'] == membership_level) & 
                           (df['quarter'] == quarter)]
        
        if membership_data.empty:
            return None
        
        # Tampilkan semua tahun yang tersedia untuk kuartal yang dipilih
        available_years = sorted(membership_data['year'].unique())
        
        # Hasil analisis per tahun
        yearly_results = []
        
        # Analisis per tahun
        for year in available_years:
            # Data untuk tahun spesifik
            year_data = membership_data[membership_data['year'] == year]
            
            # Top kategori berdasarkan frekuensi kemunculan
            category_counts = year_data['category'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            
            # Top kategori berdasarkan preferensi
            category_preferences = year_data.groupby('category')['preference_score'].mean().reset_index()
            
            # Top produk berdasarkan preferensi
            top_products = year_data.sort_values('preference_score', ascending=False).head(5)
            
            # Tambahkan hasil ke daftar
            yearly_results.append({
                'year': year,
                'quarter': quarter,
                'total_transactions': len(year_data),
                'top_categories_by_count': category_counts.head(3),
                'top_categories_by_preference': category_preferences.sort_values('preference_score', ascending=False).head(3),
                'top_products': top_products
            })
        
        return yearly_results
        
    except Exception as e:
        print(f"Error analyzing membership sales trends: {e}")
        import traceback
        traceback.print_exc()
        return None
        
def get_quarterly_product_recommendations(membership_level, quarter, df, category_mapping):
    """
    Menghasilkan rekomendasi produk untuk kuartal berikutnya berdasarkan
    tren historis kuartal sebelumnya.
    """
    try:
        # Dapatkan tren untuk kuartal yang dipilih
        quarterly_trends = analyze_membership_sales_trends(membership_level, quarter, df)
        
        if not quarterly_trends:
            return None
            
        # Ambil tahun terakhir dan tahun-tahun lain untuk perbandingan
        sorted_trends = sorted(quarterly_trends, key=lambda x: x['year'], reverse=True)
        
        # Jika hanya ada 1 tahun data, gunakan itu saja
        if len(sorted_trends) == 1:
            latest_data = sorted_trends[0]
            historical_data = []
        else:
            latest_data = sorted_trends[0]
            historical_data = sorted_trends[1:]
        
        # Kumpulkan kategori terpopuler berdasarkan hitungan dan preferensi
        popular_categories = set()
        
        # Dari tahun terakhir
        for cat_row in latest_data['top_categories_by_count'].itertuples():
            popular_categories.add(cat_row.category)
            
        for cat_row in latest_data['top_categories_by_preference'].itertuples():
            popular_categories.add(cat_row.category)
            
        # Tambahkan juga dari data historis (dengan bobot lebih rendah)
        for hist_year in historical_data:
            for cat_row in hist_year['top_categories_by_count'].head(2).itertuples():
                popular_categories.add(cat_row.category)
                
            for cat_row in hist_year['top_categories_by_preference'].head(2).itertuples():
                popular_categories.add(cat_row.category)
        
        # Buat rekomendasi berdasarkan kategori populer
        recommendations = []
        
        for category in popular_categories:
            # Filter data untuk kategori ini di kuartal yang dipilih
            category_data = df[(df['membership_level'] == membership_level) & 
                             (df['quarter'] == quarter) &
                             (df['category'] == category)]
            
            if len(category_data) > 0:
                # Ambil produk teratas berdasarkan preferensi
                top_products = category_data.sort_values('preference_score', ascending=False).head(3)
                
                # Hitung tren historis (naik/turun)
                trend_direction = "Stabil"
                trend_value = 0
                
                if len(sorted_trends) > 1:
                    # Cari kategori ini di data historis
                    current_pref = latest_data['top_categories_by_preference']
                    current_pref = current_pref[current_pref['category'] == category]['preference_score'].values
                    
                    prev_year = sorted_trends[1]
                    prev_pref = prev_year['top_categories_by_preference']
                    prev_pref = prev_pref[prev_pref['category'] == category]['preference_score'].values
                    
                    if len(current_pref) > 0 and len(prev_pref) > 0:
                        change = ((current_pref[0] - prev_pref[0]) / prev_pref[0]) * 100
                        trend_value = change
                        if change > 5:
                            trend_direction = "Naik"
                        elif change < -5:
                            trend_direction = "Turun"
                
                # Tambahkan rekomendasi untuk kategori ini
                category_name = category_mapping.get(category, f"Category {category}")
                
                # Produk rekomendasi untuk kategori ini
                product_recs = []
                for _, product in top_products.iterrows():
                    product_recs.append({
                        'product_id': product['product_id'],
                        'brand': product['brand'],
                        'price': product['price'],
                        'preference_score': product['preference_score']
                    })
                
                recommendations.append({
                    'category': category,
                    'category_name': category_name,
                    'trend_direction': trend_direction,
                    'trend_value': trend_value,
                    'avg_preference': category_data['preference_score'].mean(),
                    'transaction_count': len(category_data),
                    'recommended_products': product_recs
                })
        
        # Urutkan rekomendasi berdasarkan preferensi rata-rata
        recommendations.sort(key=lambda x: x['avg_preference'], reverse=True)
        
        return recommendations
        
    except Exception as e:
        print(f"Error generating quarterly recommendations: {e}")
        import traceback
        traceback.print_exc()
        return None
# Jalankan server
if __name__ == '__main__':
    app.run(debug=True)