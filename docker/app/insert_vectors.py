from datetime import datetime

import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

vec = VectorStore()

df = pd.read_csv("data/faq_dataset.csv", sep=";")

def prepare_record(row):
    