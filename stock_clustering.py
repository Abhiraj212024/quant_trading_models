"""
stock_clustering.py - Cluster stocks based on multiple similarity metrics
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dtaidistance import dtw
import warnings
warnings.filterwarnings('ignore')


class StockClusterer:
    def __init__(self, processed_data: Dict[str, pd.DataFrame], 
                 fundamentals: pd.DataFrame,
                 clustering_features: pd.DataFrame):
        self.processed_data = processed_data
        self.fundamentals = fundamentals
        self.clustering_features = clustering_features
        self.scaler = StandardScaler()
        self.clusters = None
        
    def create_correlation_matrix(self) -> pd.DataFrame:
        """Create return correlation matrix"""
        returns_data = {}
        for ticker, df in self.processed_data.items():
            returns_data[ticker] = df['returns']
        
        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()
    
    def dtw_distance_matrix(self, window_size: int = 252) -> pd.DataFrame:
        """
        Calculate Dynamic Time Warping distance matrix
        Uses recent price patterns for similarity
        """
        tickers = list(self.processed_data.keys())
        n = len(tickers)
        distance_matrix = np.zeros((n, n))
        
        print("Calculating DTW distances...")
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i < j:
                    # Get recent normalized prices
                    prices1 = self.processed_data[ticker1]['close'].tail(window_size).values
                    prices2 = self.processed_data[ticker2]['close'].tail(window_size).values
                    
                    # Normalize
                    prices1 = (prices1 - prices1.mean()) / prices1.std()
                    prices2 = (prices2 - prices2.mean()) / prices2.std()
                    
                    # Calculate DTW distance
                    dist = dtw.distance(prices1, prices2)
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        
        return pd.DataFrame(distance_matrix, index=tickers, columns=tickers)
    
    def feature_based_clustering(self, n_clusters: int = 5, method: str = 'kmeans') -> Dict:
        """
        Cluster stocks based on technical and statistical features
        """
        print(f"\nPerforming {method} clustering with {n_clusters} clusters...")
        
        # Prepare features
        features_df = self.clustering_features.copy()
        features_df = features_df.set_index('ticker')
        
        # Handle NaN and infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        # Standardize
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=3, min_samples=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        labels = clusterer.fit_predict(features_scaled)
        
        # Create results
        results = {
            'labels': labels,
            'tickers': features_df.index.tolist(),
            'features': features_df,
            'features_scaled': features_scaled,
            'model': clusterer
        }
        
        # Print cluster composition
        print("\nCluster Composition:")
        for cluster_id in np.unique(labels):
            cluster_tickers = [ticker for ticker, label in zip(results['tickers'], labels) 
                             if label == cluster_id]
            print(f"  Cluster {cluster_id}: {len(cluster_tickers)} stocks")
            print(f"    {', '.join(cluster_tickers[:10])}")
            if len(cluster_tickers) > 10:
                print(f"    ... and {len(cluster_tickers) - 10} more")
        
        return results
    
    def hybrid_clustering(self, n_clusters: int = 5, 
                         weights: Dict[str, float] = None) -> Dict:
        """
        Combine multiple similarity metrics for clustering
        """
        if weights is None:
            weights = {
                'features': 0.4,
                'correlation': 0.3,
                'dtw': 0.2,
                'fundamentals': 0.1
            }
        
        print("\nPerforming hybrid clustering...")
        print(f"Weights: {weights}")
        
        tickers = list(self.processed_data.keys())
        n = len(tickers)
        combined_distance = np.zeros((n, n))
        
        # 1. Feature-based distance
        if weights['features'] > 0:
            features_df = self.clustering_features.set_index('ticker')
            features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(features_df.median())
            features_scaled = StandardScaler().fit_transform(features_df)
            
            feature_dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    dist = euclidean(features_scaled[i], features_scaled[j])
                    feature_dist[i, j] = dist
                    feature_dist[j, i] = dist
            
            combined_distance += weights['features'] * self._normalize_matrix(feature_dist)
        
        # 2. Correlation-based distance
        if weights['correlation'] > 0:
            corr_matrix = self.create_correlation_matrix()
            corr_dist = 1 - corr_matrix.values  # Convert correlation to distance
            combined_distance += weights['correlation'] * self._normalize_matrix(corr_dist)
        
        # 3. DTW distance
        if weights['dtw'] > 0:
            dtw_dist = self.dtw_distance_matrix(window_size=min(252, 
                len(list(self.processed_data.values())[0])))
            combined_distance += weights['dtw'] * self._normalize_matrix(dtw_dist.values)
        
        # 4. Fundamental distance
        if weights['fundamentals'] > 0:
            fund_features = ['market_cap', 'beta', 'pe_ratio']
            fund_df = self.fundamentals[['ticker'] + fund_features].set_index('ticker')
            fund_df = fund_df.replace([np.inf, -np.inf], np.nan).fillna(fund_df.median())
            fund_scaled = StandardScaler().fit_transform(fund_df)
            
            fund_dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    dist = euclidean(fund_scaled[i], fund_scaled[j])
                    fund_dist[i, j] = dist
                    fund_dist[j, i] = dist
            
            combined_distance += weights['fundamentals'] * self._normalize_matrix(fund_dist)
        
        # Perform clustering
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, 
                                           affinity='precomputed',
                                           linkage='average')
        labels = clusterer.fit_predict(combined_distance)
        
        results = {
            'labels': labels,
            'tickers': tickers,
            'distance_matrix': combined_distance,
            'model': clusterer
        }
        
        # Print results
        print("\nHybrid Cluster Composition:")
        for cluster_id in np.unique(labels):
            cluster_tickers = [ticker for ticker, label in zip(tickers, labels) 
                             if label == cluster_id]
            cluster_sectors = self.fundamentals[
                self.fundamentals['ticker'].isin(cluster_tickers)
            ]['sector'].value_counts()
            
            print(f"\n  Cluster {cluster_id}: {len(cluster_tickers)} stocks")
            print(f"    Stocks: {', '.join(cluster_tickers[:8])}")
            if len(cluster_tickers) > 8:
                print(f"    ... and {len(cluster_tickers) - 8} more")
            print(f"    Dominant sectors: {dict(cluster_sectors.head(3))}")
        
        self.clusters = results
        return results
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize distance matrix to [0, 1]"""
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val > min_val:
            return (matrix - min_val) / (max_val - min_val)
        return matrix
    
    def visualize_clusters(self, results: Dict, save_path: str = None):
        """Visualize clustering results"""
        # PCA for 2D visualization
        if 'features_scaled' in results:
            features_scaled = results['features_scaled']
        else:
            features_df = self.clustering_features.set_index('ticker')
            features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(features_df.median())
            features_scaled = StandardScaler().fit_transform(features_df)
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=results['labels'], cmap='tab10', s=100, alpha=0.6)
        
        # Add labels
        for i, ticker in enumerate(results['tickers']):
            plt.annotate(ticker, (features_2d[i, 0], features_2d[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title('Stock Clusters (PCA Projection)')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/cluster_visualization.png", dpi=300)
        plt.show()
    
    def get_cluster_data(self, cluster_id: int) -> Dict[str, pd.DataFrame]:
        """Get data for specific cluster"""
        if self.clusters is None:
            raise ValueError("Run clustering first")
        
        cluster_tickers = [
            ticker for ticker, label in zip(self.clusters['tickers'], self.clusters['labels'])
            if label == cluster_id
        ]
        
        return {ticker: self.processed_data[ticker] for ticker in cluster_tickers}
    
    def save_clusters(self, filepath: str = 'data/clusters/'):
        """Save cluster assignments and data"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        if self.clusters is None:
            raise ValueError("Run clustering first")
        
        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'ticker': self.clusters['tickers'],
            'cluster': self.clusters['labels']
        })
        
        # Add sector information
        cluster_df = cluster_df.merge(
            self.fundamentals[['ticker', 'sector', 'industry']], 
            on='ticker', 
            how='left'
        )
        
        cluster_df.to_csv(f"{filepath}cluster_assignments.csv", index=False)
        
        # Save each cluster's data separately
        for cluster_id in np.unique(self.clusters['labels']):
            cluster_data = self.get_cluster_data(cluster_id)
            cluster_dir = f"{filepath}cluster_{cluster_id}/"
            os.makedirs(cluster_dir, exist_ok=True)
            
            for ticker, df in cluster_data.items():
                df.to_csv(f"{cluster_dir}{ticker}.csv")
        
        print(f"\n✓ Clusters saved to {filepath}")
        print(f"  - Cluster assignments")
        print(f"  - {len(np.unique(self.clusters['labels']))} cluster directories")


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load data
    data_path = 'data/'
    processed_data = {}
    
    for file in os.listdir(data_path):
        if file.endswith('_features.csv'):
            ticker = file.replace('_features.csv', '')
            processed_data[ticker] = pd.read_csv(f"{data_path}{file}", index_col=0, parse_dates=True)
    
    fundamentals = pd.read_csv(f"{data_path}fundamentals.csv")
    clustering_features = pd.read_csv(f"{data_path}clustering_features.csv")
    
    # Create clusterer
    clusterer = StockClusterer(processed_data, fundamentals, clustering_features)
    
    # Perform hybrid clustering
    results = clusterer.hybrid_clustering(n_clusters=5)
    
    # Visualize
    clusterer.visualize_clusters(results, save_path=data_path)
    
    # Save
    clusterer.save_clusters()