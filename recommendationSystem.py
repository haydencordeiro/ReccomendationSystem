import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from gensim.models import Word2Vec
import pandas as pd
from fastapi.responses import JSONResponse
import pickle

app = FastAPI()


def embedding_query(query):
    model = Word2Vec.load("./recomModel.model")
    print("tes", 'blue' in model.wv)
    print("before",query)
    query = str(query).split(" ")
    query = [i.replace('"',"") for i in query]
    print(query)
    query_vec = np.zeros(100).reshape((1, 100))
    count = 0
    for word in query:
        print(word in model.wv)
        if word in model.wv:
            print("word", word, " ", model.wv[word])
            query_vec += model.wv[word]
            count += 1.
    if count != 0:
        query_vec /= count
    return query_vec


def get_similarity(query, n_top, df):
    print(query)
    query_vec = embedding_query(query)
    # df['cleaned_embedding'] = df['cleaned_embedding'].s
    # df['cleaned_embedding'] = df['cleaned_embedding'].apply(lambda arr: [float(x) for x in arr if x not in [] ])
    print(query_vec)

    print(df['cleaned_embedding'])
    df["cos_sim"] = df['cleaned_embedding'].apply(
        lambda x: cosine_similarity(
            [x], query_vec.reshape(1, -1))[0][0])
    print(df["cos_sim"])
    top_list = (df.sort_values("cos_sim", ascending=False)
                [["product_id"]]
                .drop_duplicates()[:n_top])
    return top_list


@app.post("/recommendation/")
def getRecommendation(text):
    text = str(text)
    with open("embeddings_vector.pkl", 'rb') as file:
        embedding_df = pickle.load(file)

    recom_products = get_similarity(text, 10, embedding_df)
    total_data = pd.read_csv("data.csv")
    jsonResponse = []

    for index, row in recom_products.iterrows():
        product_id = row['product_id']
        product = total_data[total_data["product_id"] == product_id].iloc[0]
        jsonObj = {
            "productId": str(row['product_id']),
            "productName": str(product['title']),
            "productPrice": str(product['variant_price']),
            "productImageUrl": str(product['images'])
        }
        jsonResponse.append(jsonObj)
    # json_compatible_item_data = jsonable_encoder({"data": jsonResponse})
    # print(jsonResponse)
    return JSONResponse(content={"data": jsonResponse})
