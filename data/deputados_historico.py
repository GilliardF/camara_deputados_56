import pandas as pd
import sqlite3


def deputados_historico(id_deputado, db_path):
    query = """
    SELECT
        siglaPartido,
        dataHora,
        situacao,
        condicaoEleitoral,
        descricaoStatus 
    FROM
        deputados_56_partidos dp
    WHERE
        dp.id_deputado = ?
    ORDER BY
        dp.dataHora ASC;
    """

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(query, conn, params=(id_deputado,))
