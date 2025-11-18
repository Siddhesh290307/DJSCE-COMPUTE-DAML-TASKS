import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.express as px
import streamlit as st
import hdbscan  

#Generating dataset
def generate_dataset(dataset_type, n_samples, n_features=2, n_clusters_true=3, noise=0.1):
    if dataset_type == "Blobs":
        X, y_true = make_blobs(n_samples=n_samples, n_features=n_features,
                               centers=n_clusters_true, cluster_std=1.0, random_state=42)
    else:
        X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    return X, df


#Scaling data as per what the user wants
def scale_data(X, scaler_option):
    if scaler_option == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_option == "RobustScaler":
        scaler = RobustScaler()
    else:
        return X.copy()
    return scaler.fit_transform(X)


#Clustering dataset
def cluster_and_plot(dataset_type, n_samples, scaler_option, algorithm,
                     k=3, eps=0.5, min_samples=5, n_clusters=3,
                     linkage='ward', min_cluster_size=10, n_features=2, noise=0.1):

    # Generate and scale dataset
    X, df = generate_dataset(dataset_type, n_samples, n_features, n_clusters, noise)
    X_scaled = scale_data(X, scaler_option)

    # Choose algorithm
    if algorithm == "K-Means":
        model = KMeans(n_clusters=k, random_state=42)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    elif algorithm == "HDBSCAN":
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

    # Fit model
    labels = model.fit_predict(X_scaled)

    # Compute Silhouette Score safely
    unique_labels = set(labels)
    n_clusters_computed = len(unique_labels) - (1 if -1 in unique_labels else 0)
    if n_clusters_computed > 1:
        valid_indices = labels != -1
        score = silhouette_score(X_scaled[valid_indices], labels[valid_indices])
        st.write(f"**Silhouette Score:** {score:.3f}")
    else:
        st.write("⚠️ Silhouette Score cannot be computed (single cluster or only noise points).")

    # Plot
    if X_scaled.shape[1] == 2:
        fig = px.scatter(df, x="Feature 1", y="Feature 2", color=labels.astype(str),
                         title=f"{algorithm} Clustering Results", labels={"color": "Cluster"})
    else:
        fig = px.scatter_3d(df, x="Feature 1", y="Feature 2", z="Feature 3",
                            color=labels.astype(str), title=f"{algorithm} Clustering Results",
                            labels={"color": "Cluster"})
    st.plotly_chart(fig)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 Interactive Clustering Visualization")

dataset_type = st.selectbox("Dataset Type", ["Blobs", "Moons"])
n_samples = st.slider("Number of Samples", 500, 2000, 1000, 100)
scaler_option = st.selectbox("Scaler", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
algorithm = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Agglomerative", "HDBSCAN"])
n_features = st.slider("Number of Features (for Blobs)", 2, 3, 2)
noise = st.slider("Noise (for Moons)", 0.0, 0.5, 0.1, 0.05)

if algorithm == "K-Means":
    k = st.slider("k (K-Means)", 2, 10, 3)
    cluster_and_plot(dataset_type, n_samples, scaler_option, algorithm, k=k, n_features=n_features, noise=noise)

elif algorithm == "DBSCAN":
    eps = st.slider("eps (DBSCAN)", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.slider("min_samples (DBSCAN)", 2, 20, 5)
    cluster_and_plot(dataset_type, n_samples, scaler_option, algorithm,
                     eps=eps, min_samples=min_samples, n_features=n_features, noise=noise)

elif algorithm == "Agglomerative":
    n_clusters = st.slider("Clusters (Agglomerative)", 2, 10, 3)
    linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
    cluster_and_plot(dataset_type, n_samples, scaler_option, algorithm,
                     n_clusters=n_clusters, linkage=linkage, n_features=n_features, noise=noise)

elif algorithm == "HDBSCAN":
    min_cluster_size = st.slider("min_cluster_size (HDBSCAN)", 5, 50, 10)
    cluster_and_plot(dataset_type, n_samples, scaler_option, algorithm,
                     min_cluster_size=min_cluster_size, n_features=n_features, noise=noise)
