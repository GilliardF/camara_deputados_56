import pandas as pd
import sqlite3

def distribuicao_uf(sigla_partido, db_path):
    query = """
        SELECT
            d.siglaUf,
            COUNT(DISTINCT d.id) AS total_deputados
        FROM
            deputados_56 d
        INNER JOIN
            partidos p ON d.siglaPartido = p.sigla
        WHERE
            d.siglaPartido = :partido
        GROUP BY
            d.siglaUf;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(query, conn, params={"partido": sigla_partido})
