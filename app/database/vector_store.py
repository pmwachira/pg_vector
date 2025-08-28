import logging
import time
from typing import List, Any, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
from config.settings import get_settings
from openai import OpenAI
from timescale_vector import client


class VectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model
        self.vector_settings = self.settings.vector_store
        self.vec_client = client.Sync(
            self.settings.database.service_url,
            self.vector_settings.table_name,
            self.vector_settings.table_name,
            self.vector_settings.embedding_dimension,
            time_partition_interval = self.vector_settings.time_partition_interval
        )
    

    def create_tables(self) -> None:
        self.vec_client.create_table()

    def create_index(self) -> None:
        self.vec_client.create_embedding_index(client.DiskAnnIndex())

    def drop_index(self) -> None:
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings.table_name}"
        )

    def search(
            self,
            query_text: str,
            limit: int = 5,
            metadata_filter: Union[dict, List[dict]] = None,
            predicates: Optional[client.Predicates] = None,
            time_range: Optional[Tuple[datetime, datetime]] = None,
            return_dataframe: bool = True
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        query_embedding = self.get_embedding(query_text)
        start_time = time.time()
        search_args ={
            "limit": limit
        }

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates
        
        if time_range:
            start_date, end_date = time_range   
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.vec_client.search(query_embedding, **search_args)

        elapsed_time = time.time() - start_time
        logging.info(f"Search completed in {elapsed_time:.2f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results
        
    def _create_dataframe_from_results(
            self,
            results: List[Tuple[Any, ...]]
    )-> pd.DataFrame:
        columns = ["id", "metadata", "content", "embedding", "distance"]
        df = pd.DataFrame(results, columns=columns)

        df = pd.concat(
            [df.drop(["metadata"],axis=1),df["metadata"].apply(pd.Series)],axis = 1
        )
        df[id]= df["id"].astype(str)

        return df
    
    def delete(
            self,
            ids: List[str]= None,
            metadata_filter: dict = None,
            delete_all: bool = False
    )-> None:

        if sum(bool(x) for x in [ids, metadata_filter, delete_all]) != 1:
            raise ValueError("Provide exactly one of ids, metadata_filter, or delete_all")
        

        if delete_all:
            self.vec_client.delete_all()
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")

        elif ids:
            self.vec_client.delete_by_ids(ids=ids)
            logging.info(f"Deleted {len(ids)} records from {self.vector_settings.table_name}")

        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info(f"Deleted records matching filter {metadata_filter} from {self.vector_settings.table_name}")

    def get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = (
            self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )
            .data[0]
            .embedding
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding created in {elapsed_time:.2f} seconds")
        
        return embedding