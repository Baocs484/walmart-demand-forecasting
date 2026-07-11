# src/store_clustering.py - SỬA LẠI
"""
Store clustering với similar store detection
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from .utils import logger

class StoreClustering:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.store_clusters = {}
        self.store_profiles = None
        self.cluster_centers = None
    
    def fit(self, df):
        """Phân cụm stores với nhiều tiêu chí"""
        logger.info("Clustering stores...")
        
        # ========== SỬA LỖI: Tính Holiday_Lift AN TOÀN ==========
        
        # Tạo store profiles
        profiles = df.groupby('Store').agg({
            'Weekly_Sales': ['mean', 'std', 'sum'],
            'Size': 'first',
            'Type': 'first',
            'Temperature': 'mean',
            'Fuel_Price': 'mean'
        }).reset_index()
        
        profiles.columns = ['Store', 'Sales_mean', 'Sales_std', 'Sales_sum', 
                           'Size', 'Type', 'Temp', 'Fuel']
        
        # ✨ TÍNH HOLIDAY_LIFT RIÊNG (an toàn hơn)
        holiday_lifts = []
        
        for store_id in profiles['Store']:
            store_data = df[df['Store'] == store_id]
            
            # Tính trung bình sales cho holiday và non-holiday
            holiday_sales = store_data[store_data['IsHoliday'] == True]['Weekly_Sales'].mean()
            normal_sales = store_data[store_data['IsHoliday'] == False]['Weekly_Sales'].mean()
            
            # An toàn: Check NaN và division by zero
            if pd.notna(holiday_sales) and pd.notna(normal_sales) and normal_sales > 0:
                lift = holiday_sales / normal_sales
            else:
                lift = 1.0  # Default: no lift
            
            holiday_lifts.append(lift)
        
        profiles['Holiday_Lift'] = holiday_lifts
        
        # ========== TYPE ENCODING ==========
        profiles['Type_encoded'] = profiles['Type'].map({'A': 2, 'B': 1, 'C': 0}).fillna(1)
        
        # ========== VOLATILITY (CV) ==========
        profiles['Volatility_CV'] = profiles['Sales_std'] / (profiles['Sales_mean'] + 1e-6)
        
        # ========== FEATURES FOR CLUSTERING ==========
        feature_cols = ['Sales_mean', 'Sales_std', 'Size', 'Type_encoded', 
                       'Temp', 'Fuel', 'Holiday_Lift', 'Volatility_CV']
        
        X = self.scaler.fit_transform(profiles[feature_cols].fillna(0))
        
        # ========== KMEANS ==========
        n_eff = min(self.n_clusters, len(profiles))
        
        if n_eff == 1:
            clusters = np.zeros(len(profiles), dtype=int)
            self.cluster_centers = X.mean(axis=0).reshape(1, -1)
        else:
            kmeans = KMeans(n_clusters=n_eff, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            self.cluster_centers = kmeans.cluster_centers_
        
        profiles['Cluster'] = clusters
        self.store_profiles = profiles
        self.store_clusters = dict(zip(profiles['Store'], profiles['Cluster']))
        
        # ========== LOG INFO ==========
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        logger.info(f"✓ Clustered into {n_eff} groups")
        
        for cluster_id, count in cluster_counts.items():
            cluster_stores = profiles[profiles['Cluster'] == cluster_id]
            avg_sales = cluster_stores['Sales_mean'].mean()
            avg_lift = cluster_stores['Holiday_Lift'].mean()
            logger.info(f"  Cluster {cluster_id}: {count} stores, avg_sales={avg_sales:.0f}, holiday_lift={avg_lift:.2f}")
        
        return self.store_clusters
    
    def find_similar_stores(self, store_id, top_k=3):
        """Tìm stores tương tự (for transfer learning)"""
        if self.store_profiles is None:
            return []
        
        store_row = self.store_profiles[self.store_profiles['Store'] == store_id]
        if store_row.empty:
            return []
        
        cluster = self.store_clusters[store_id]
        candidates = self.store_profiles[
            (self.store_profiles['Cluster'] == cluster) & 
            (self.store_profiles['Store'] != store_id)
        ]
        
        if candidates.empty:
            return []
        
        # Cosine similarity
        feature_cols = ['Sales_mean', 'Sales_std', 'Size', 'Holiday_Lift']
        store_vec = store_row[feature_cols].values[0]
        cand_vecs = candidates[feature_cols].values
        
        # Normalize vectors
        store_vec_norm = store_vec / (np.linalg.norm(store_vec) + 1e-6)
        cand_vecs_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-6)
        
        # Dot product
        similarities = np.dot(cand_vecs_norm, store_vec_norm)
        
        top_idx = np.argsort(similarities)[::-1][:top_k]
        return candidates.iloc[top_idx]['Store'].tolist()
    
    def get_cluster(self, store_id):
        return self.store_clusters.get(store_id, 0)
    
    def get_cluster_statistics(self, cluster_id):
        """Lấy thống kê của một cluster"""
        if self.store_profiles is None:
            return {}
        
        cluster_stores = self.store_profiles[self.store_profiles['Cluster'] == cluster_id]
        
        if cluster_stores.empty:
            return {}
        
        stats = {
            'n_stores': len(cluster_stores),
            'avg_sales': cluster_stores['Sales_mean'].mean(),
            'total_sales': cluster_stores['Sales_sum'].sum(),
            'avg_size': cluster_stores['Size'].mean(),
            'avg_volatility': cluster_stores['Volatility_CV'].mean(),
            'avg_holiday_lift': cluster_stores['Holiday_Lift'].mean()
        }
        
        return stats