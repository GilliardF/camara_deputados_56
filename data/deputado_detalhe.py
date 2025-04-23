import pandas as pd
import sqlite3


def deputado_detalhe(select_deputado, db_path):
    query = """
        SELECT
        d_56.id,
        d_56.nome,
        d_56_detalhes.nomeCivil,
        d_56.siglaUf,
        d_56.urlFoto,
        d_56_detalhes.cpf,
        d_56_detalhes.dataNascimento,
        d_56_detalhes.dataFalecimento,
        d_56_detalhes.escolaridade,
        d_56_detalhes.profissoes,
        d_56_detalhes.redeSocial,
        d_56.siglaPartido,
        partidos.nome AS nomePartido,
        partidos.urlLogo
    FROM
        deputados_56 AS d_56
    JOIN
        deputados_56_detalhes AS d_56_detalhes
        ON d_56.id = d_56_detalhes.id
    JOIN
        partidos
        ON d_56.siglaPartido = partidos.sigla
    WHERE
        d_56.nome = ?;
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(query, conn, params=(select_deputado,))
