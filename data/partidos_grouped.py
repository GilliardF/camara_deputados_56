import sqlite3
import pandas as pd


def partidos_grouped(sigla_partido, db_path):
    query = """
        SELECT
            siglaUf,
            COUNT(id) AS total_deputados
        FROM
            deputados_56
        WHERE
            siglaPartido = ?
        GROUP BY
            siglaUf;
    """

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(query, conn, params=(sigla_partido,))
