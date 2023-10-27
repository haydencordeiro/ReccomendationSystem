
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from gensim.models import Word2Vec
import pandas as pd
from fastapi.responses import JSONResponse
import pickle

app = FastAPI()

def embedding_query(query, model):
    query = query.split(' ')
    query_vec = np.zeros(100).reshape((1,100))
    count = 0
    for word in query:
        if word in model.wv:
            query_vec += model.wv[word]
            count += 1.
    if count != 0:
        query_vec /= count
    return query_vec


def get_similarity(query, n_top, df, model):
    query_vec = embedding_query(query, model)
    # df['cleaned_embedding'] = df['cleaned_embedding'].s
    # df['cleaned_embedding'] = df['cleaned_embedding'].apply(lambda arr: [float(x) for x in arr if x not in [] ])
    df["cos_sim"] = df['cleaned_embedding'].apply(
        lambda x: cosine_similarity(
            [x], query_vec.reshape(1, -1))[0][0])
    top_list = (df.sort_values("cos_sim", ascending=False)
                [["product_id"]]
                .drop_duplicates()[:n_top])
    return top_list

@app.post("/recommendation/")
def getRecommendation(text):
    recom_model = Word2Vec.load("./recomModel.model")
    with open("./embeddings_vector.pkl", 'rb') as file:
        embedding_df = pickle.load(file)

    recom_products = get_similarity(text, 10, embedding_df, recom_model)
    total_data = pd.read_csv("./data.csv")
    jsonResponse = []

    for index, row in recom_products.iterrows():
        product_id = row['product_id']
        product = total_data[total_data["product_id"]==product_id].iloc[0]
        jsonObj = {
            "productId": str(row['product_id']),
            "productTitle": str(product['title']),
            "productPrice": str(product['variant_price']),
            "productImageUrl": str(product['images'])
        }
        jsonResponse.append(jsonObj)
    # json_compatible_item_data = jsonable_encoder({"data": jsonResponse})
    # print(jsonResponse)
    return JSONResponse(content={"data": jsonResponse})

