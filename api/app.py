from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # Configura o CORS para aceitar requisições de qualquer origem

with open('api\knn_model1.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

products_pivot = pd.read_csv('api\products_data.csv', index_col=0)

@app.route('/recommend', methods=['POST'])  
def recommend():
    data = request.json
    product_name = data['product_name']

    if product_name not in products_pivot.index:
        return jsonify({'error': 'Produto não encontrado'}), 404

    product_idx = products_pivot.index.get_loc(product_name)

    distances, indices = knn_model.kneighbors([products_pivot.iloc[product_idx].values])
    
    recommended_products = products_pivot.index[indices.flatten()].tolist()

    return jsonify({'recommendations': recommended_products})

if __name__ == '__main__':
    app.run(debug=True)
