import sqlite3
import pandas as pd


def despesas_deputado(id_deputado, db_path):
    query = """
        SELECT
            *
        FROM
            despesas d
        WHERE
            d.id_deputado = :id_deputado;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(query, conn, params={"id_deputado": id_deputado})
