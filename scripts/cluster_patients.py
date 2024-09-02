import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, dataset):
        self.dataset = dataset

    def encode_categorical(self):
        label_encoders = {}
        for column in self.dataset.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.dataset[column] = le.fit_transform(self.dataset[column])
            label_encoders[column] = le
        return self.dataset, label_encoders

    def standardize_data(self):
        scaler = StandardScaler()
        return scaler.fit_transform(self.dataset)

def elbow_method(data, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

class ClusteringModel:
    def __init__(self, n_clusters=4):
        self.model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)

    def fit_predict(self, data):
        return self.model.fit_predict(data)

class Visualizer:
    @staticmethod
    def plot_elbow(wcss):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show()

class ClusterAnalyzer:
    @staticmethod
    def analyze_clusters(dataset, cluster_labels):
        dataset['Cluster'] = cluster_labels
        return dataset.groupby('Cluster').mean()

# Veri Yükleme
dataset = pd.read_csv('data\\drug200.csv')

# Adım 1: Veriyi İşleme
preprocessor = DataPreprocessor(dataset)
processed_dataset, label_encoders = preprocessor.encode_categorical()
X_scaled = preprocessor.standardize_data()

# Adım 2: Elbow Method'u Uygulama
wcss = elbow_method(X_scaled, max_clusters=10)

# Dirsek Yöntemini Görselleştirme
Visualizer.plot_elbow(wcss)

# Adım 3: K-Means Kümeleme Modeli Uygulama
clustering_model = ClusteringModel(n_clusters=4)
clusters = clustering_model.fit_predict(X_scaled)

# Adım 4: Kümeleri Analiz Etme
cluster_analysis = ClusterAnalyzer.analyze_clusters(processed_dataset, clusters)

# Analiz Sonucunu Görüntüleme
print(cluster_analysis)
