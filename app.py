import json
import pandas as pd
import datetime as dt
from flask import Flask, request, jsonify, render_template, redirect, url_for
from pymongo import MongoClient
from clustering import panggil_function
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Menghubungkan ke MongoDB
uri = MongoClient('mongodb+srv://triskawidiantari:Toritralala@cluster0.oxfsbj9.mongodb.net/')
db_name = uri['dav_final']
collection = db_name['data']
collection2 = db_name['data_hasil']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get-level', methods = ['GET'])
def get_level():
    try:
        pipeline = [
            {"$group": {"_id": "$RFM_Loyalty_Level", "count": {"$sum": 1}}}
        ]

        loyalty_levels_distribution = list(collection2.aggregate(pipeline))

        formatted_response = [{"level": item["_id"], "count": item["count"]} for item in loyalty_levels_distribution]

        response = {
            "loyalty_levels": formatted_response
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/customer-loyalty', methods=['GET'])
def get_customers_by_loyalty_level():
    try:
        pipeline = [
            {"$group": {"_id": "$RFM_Loyalty_Level", "Customer": {"$push": "$Customer"}}}
        ]
        loyalty_levels_customers = list(collection2.aggregate(pipeline))

        response = {}
        for level_customers in loyalty_levels_customers:
            level = level_customers['_id']
            customers = level_customers['Customer']
            response[level] = customers

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

# Fetch data from MongoDB and prepare for clustering
def fetch_data():
    data = list(collection2.find({}, {'_id': 0, 'Recency': 1, 'Frequency': 1, 'Monetary': 1}))
    df = pd.DataFrame(data)
    X = df[['Recency', 'Frequency', 'Monetary']]
    X = StandardScaler().fit_transform(X)
    return X

@app.route('/scatter-plot', methods=['GET'])
def scatter_plot():
    try:
        # Get clustering method from query parameters
        method = request.args.get('method', 'kmeans')
        X = fetch_data()
        
        # Perform clustering
        if method == 'dbscan':
            eps = float(request.args.get('eps', 0.3))
            min_samples = int(request.args.get('min_samples', 5))
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clustering.fit_predict(X)
        else:
            n_clusters = int(request.args.get('n_clusters', 3))
            clustering = KMeans(n_clusters=n_clusters)
            labels = clustering.fit_predict(X)

        # Prepare data for scatter plot
        df = pd.DataFrame(X, columns=['x', 'y', 'z'])
        df['labels'] = labels

        # Convert data to JSON format
        scatter_data = df.to_dict(orient='records')
        return jsonify(scatter_data)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/submit', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.json
            date_str = data['date']
            customer = data['customer']
            raspberry = int(data['raspberry'])
            blueberry = int(data['blueberry'])
            frozen_raspberry = int(data['frozen_raspberry'])
            cape_gooseberry = int(data['cape_gooseberry'])
            tea = int(data['tea'])
            jam = int(data['jam'])
            no_nota = data['no_nota']
            price_per_grs = float(data['price_per_grs'])
        else:
            date_str = request.form['date']
            customer = request.form['customer']
            raspberry = int(request.form['raspberry'])
            blueberry = int(request.form['blueberry'])
            frozen_raspberry = int(request.form['frozen_raspberry'])
            cape_gooseberry = int(request.form['cape_gooseberry'])
            tea = int(request.form['tea'])
            jam = int(request.form['jam'])
            no_nota = request.form['no_nota']
            price_per_grs = float(request.form['price_per_grs'])

        total_harga = (raspberry + blueberry + frozen_raspberry + cape_gooseberry + tea + jam) * price_per_grs
        date = dt.datetime.strptime(date_str, "%Y-%m-%d")

        new_data = {
            "DATE": date,
            "CUSTOMER": customer,
            "RASPBERRY": raspberry,
            "BLUEBERRY": blueberry,
            "FROZEN RASPBERRY": frozen_raspberry,
            "CAPE GOOSEBERRY": cape_gooseberry,
            "TEA": tea,
            "JAM": jam,
            "NO NOTA": no_nota,
            "PRICE /GRS": price_per_grs,
            "TOTAL HARGA": total_harga
        }

        collection.insert_one(new_data)

        data = list(collection.find())
        new_df = pd.DataFrame(data)

        new_hasil = panggil_function(new_df, n_clusters=3, eps=1, min_samples=5)

        result_df = cari_berdasarkan_nama(new_hasil, customer)

        result_json = result_df.to_json(orient='records')
        documents = result_df.to_dict(orient='records')
    
        collection2.insert_many(documents)

        return jsonify(json.loads(result_json))

    except Exception as e:
        return jsonify({"error": str(e)})


def cari_berdasarkan_nama(df, nama_dicari):
    return df[df['Customer'] == nama_dicari]


if __name__ == '__main__':
    app.run(debug=True)
