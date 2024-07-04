import pandas as pd
import datetime as dt
import numpy as np
import pymongo
from pymongo import MongoClient

from random import sample
from numpy.random import uniform

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import KElbowVisualizer


# Fungsi untuk menghubungkan ke MongoDB
def connect_to_mongodb(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection

# Hubungkan ke MongoDB
uri = 'mongodb+srv://triskawidiantari:Toritralala@cluster0.oxfsbj9.mongodb.net/'
db_name = 'dav_terbaru'
collection_name = 'data'
collection = connect_to_mongodb(uri, db_name, collection_name)

# RFM Analysis
def rfm_analysis(df, latest_date):
    # Membuat RFM Modelling scores untuk setiap customer
    rfm_scores = df.groupby('CUSTOMER').agg({
        'DATE': lambda x: (latest_date - x.max()).days,
        'NO NOTA': lambda x: len(x),
        'TOTAL HARGA': lambda x: x.sum()
    })
    
    # Convert data type dari DATE menjadi int
    rfm_scores['DATE'] = rfm_scores['DATE'].astype(int)
    
    # Rename columns pada rfm_scores
    rfm_scores.rename(columns={
        'DATE': 'Recency',
        'NO NOTA': 'Frequency',
        'TOTAL HARGA': 'Monetary'
    }, inplace=True)
    
    # Split menjadi empat segmen menggunakan kuartil
    split = rfm_scores.quantile(q=[0.25, 0.5, 0.75]).to_dict()
    
    # Fungsi untuk membuat R, F, dan M segments
    def RScoring(x, p, d):
        if x <= d[p][0.25]: return 1
        elif x <= d[p][0.50]: return 2
        elif x <= d[p][0.75]: return 3
        else: return 4

    def FScoring(x, p, d):
        if x <= d[p][0.25]: return 4
        elif x <= d[p][0.50]: return 3
        elif x <= d[p][0.75]: return 2
        else: return 1

    def MScoring(x, p, d):
        if x <= d[p][0.25]: return 4
        elif x <= d[p][0.50]: return 3
        elif x <= d[p][0.75]: return 2
        else: return 1

    # Hitung & Tambahkan kolom nilai segmen R, F, dan M
    rfm_scores['R'] = rfm_scores['Recency'].apply(RScoring, args=('Recency', split))
    rfm_scores['F'] = rfm_scores['Frequency'].apply(FScoring, args=('Frequency', split))
    rfm_scores['M'] = rfm_scores['Monetary'].apply(MScoring, args=('Monetary', split))
    
    # Hitung dan tambahkan pada kolom baru RFMGroup
    rfm_scores['RFMGroup'] = rfm_scores.R.map(str) + rfm_scores.F.map(str) + rfm_scores.M.map(str)
    
    # Hitung dan tambahkan pada kolom baru RFMScore
    rfm_scores['RFMScore'] = rfm_scores[['R', 'F', 'M']].sum(axis=1)
    
    # Menetapkan Loyalty Level
    Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze']
    Score_cuts = pd.qcut(rfm_scores.RFMScore, q = 4, labels = Loyalty_Level)
    rfm_scores['RFM_Loyalty_Level'] = Score_cuts.values
    
    return rfm_scores.reset_index()

# Clustering K-Means
# Fungsi untuk menangani nilai negatif atau nol selama transformasi log
def handle_negative_values(num):
    if num <= 0:
        return 1
    else:
        return num

# Fungsi untuk transformasi log dan normalisasi data
def log_and_normalize_rfm(rfm_scores):
    rfm_scores['Recency'] = [handle_negative_values(x) for x in rfm_scores.Recency]
    rfm_scores['Frequency'] = [handle_negative_values(x) for x in rfm_scores.Frequency]
    rfm_scores['Monetary'] = [handle_negative_values(x) for x in rfm_scores.Monetary]

    Log_RFM = rfm_scores[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)

    # Normalisasi values
    scaler = StandardScaler()
    scaler.fit(Log_RFM)
    scaled_RFM = pd.DataFrame(scaler.transform(Log_RFM), columns= Log_RFM.columns)
    
    return scaled_RFM

# Fungsi untuk menentukan elbow point dan visualisasinya
def find_elbow_point(scaled_RFM):
    model = KMeans(random_state=42)
    visualizer = KElbowVisualizer(model, k=(1,10))
    visualizer.fit(scaled_RFM)
    visualizer.show()

# Fungsi untuk clustering menggunakan KMeans
def kmeans_clustering(scaled_RFM, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_RFM)
    scaled_RFM['KMeans_Cluster'] = clusters
    return scaled_RFM

# Clustering DBSCAN
# Fungsi untuk clustering menggunakan DBSCAN
def dbscan_clustering(scaled_RFM, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_RFM)
    scaled_RFM['DBSCAN_Cluster'] = clusters
    return scaled_RFM


# Panggil Function
# Mengambil data dari MongoDB dan mengubahnya menjadi DataFrame
data = list(collection.find())
df_clean = pd.DataFrame(data)

def panggil_function(df, n_clusters=None, eps=None, min_samples=None):
    latest_date = dt.datetime(2023, 11, 30)

    # Proses RFM Analysis
    rfm_hasil = rfm_analysis(df, latest_date)
    
    # Transformasi log dan normalisasi
    scaled_RFM = log_and_normalize_rfm(rfm_hasil)
    
    # Simpan kolom Loyalty Level sebelum transformasi
    customer = rfm_hasil['CUSTOMER']
    loyalty_level = rfm_hasil['RFM_Loyalty_Level']
    
    
    # Melakukan clustering menggunakan KMeans dengan n_clusters optimal
    kmeans = KMeans(n_clusters=n_clusters)
    hasil_kmeans = kmeans.fit_predict(scaled_RFM)

    # Melakukan clustering menggunakan DBSCAN dengan parameter yang telah ditentukan
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    hasil_dbscan = dbscan.fit_predict(scaled_RFM)

    
    # Tambahkan kembali kolom Loyalty Level
    hasil_df = pd.DataFrame({
        'Customer': customer,
        'Recency': rfm_hasil['Recency'],
        'Frequency': rfm_hasil['Frequency'],
        'Monetary': rfm_hasil['Monetary'],
        'R Score': rfm_hasil['R'],
        'F Score': rfm_hasil['F'],
        'M Score': rfm_hasil['M'],
        'RFM_Loyalty_Level': loyalty_level,
        'Cluster_Label_Kmeans': hasil_kmeans,
        'Cluster_Label_DBSCAN': hasil_dbscan
    })
    
    return hasil_df
    

hasil = panggil_function(df_clean, n_clusters=3, eps=1, min_samples=5)


# # Hubungkan ke MongoDB untuk save data hasil
# collection_name_save = 'data_hasil'
# collection_save = connect_to_mongodb(uri, db_name, collection_name_save)

# # Fungsi untuk menyimpan atau memperbarui data ke MongoDB
# def save_to_mongodb(collection, data):
#     for record in data:
#         # Periksa apakah data pelanggan sudah ada berdasarkan Customer ID
#         existing_record = collection.find_one({'Customer': record['Customer']})
#         if existing_record:
#             # Update data pelanggan jika sudah ada
#             collection.update_one({'Customer': record['Customer']}, {'$set': record})
#         else:
#             # Insert data pelanggan baru
#             collection.insert_one(record)

# # Mengubah DataFrame ke format dictionary dan menyimpan ke MongoDB
# data_dict = hasil.to_dict("records")
# save_to_mongodb(collection_save, data_dict)

