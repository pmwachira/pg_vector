from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

vec = VectorStore()

df = pd.read_csv("data/faq_dataset.csv", sep=";")   

def prepare_record(row):

    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec.get_embedding(content)
    return pd.Series({
        "id": uuid_from_time(datetime.now()),
         "metadata":{
             "category":row['category'],
             "created_at": datetime.now()
         },
         "contents": content,
         "embedding": embedding
         })

records_df = df.apply(prepare_record, axis=1)

vec.create_tables()
vec.create_index()
vec.upsert(records_df)