import sqlite3
import pandas as pd


def ML_despesas(db_path):
    query = """
        WITH geral_deputados AS (
            SELECT DISTINCT
                id,
                nome,
                siglaUf
            FROM deputados_56
        )
        SELECT
            d.id_deputado,
            gd.nome,
            gd.siglaUf,
            d.dataDocumento,
            d.tipoDespesa,
            d.valorDocumento,
            d.valorLiquido,
            d.codDocumento
        FROM despesas d
        INNER JOIN geral_deputados gd
            ON d.id_deputado = gd.id
        ORDER BY gd.nome ASC, d.dataDocumento ASC
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(query, conn)
