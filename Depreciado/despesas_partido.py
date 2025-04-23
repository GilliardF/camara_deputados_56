import sqlite3
import pandas as pd


def despesas_deputado(sigla_partido, db_path):
    query = """
        WITH partido_periodo AS (
            SELECT
                deputados_56_partidos.id_deputado,
                deputados_56.nome,
                deputados_56.urlFoto,
                deputados_56.siglaUf,
                deputados_56_partidos.siglaPartido,
                partidos.nome AS nome_partido,
                deputados_56_partidos.dataHora AS dataEntrada,
                LEAD(deputados_56_partidos.dataHora, 1, '2099-12-31')
                    OVER (PARTITION BY deputados_56_partidos.id_deputado ORDER BY deputados_56_partidos.dataHora) AS dataSaida
            FROM deputados_56_partidos
            JOIN deputados_56 ON deputados_56_partidos.id_deputado = deputados_56.id
            JOIN partidos ON deputados_56_partidos.siglaPartido = partidos.sigla
            WHERE deputados_56_partidos.descricaoStatus LIKE 'Entrada%'
        )

        SELECT
            despesas.id_deputado,
            partido_periodo.nome,
            partido_periodo.urlFoto,
            partido_periodo.siglaUf,
            partido_periodo.siglaPartido,
            partido_periodo.nome_partido,
            strftime('%Y-%m', despesas.dataDocumento) AS mes_ano,
            SUM(despesas.valorDocumento) AS total_gasto
        FROM despesas
        JOIN partido_periodo
            ON despesas.id_deputado = partido_periodo.id_deputado
            AND despesas.dataDocumento BETWEEN partido_periodo.dataEntrada AND partido_periodo.dataSaida
        WHERE partido_periodo.siglaPartido = :partido -- Adicionando um filtro por partido
        GROUP BY despesas.id_deputado, partido_periodo.nome, partido_periodo.siglaPartido, partido_periodo.nome_partido, mes_ano
        ORDER BY despesas.id_deputado, mes_ano;
    """

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(query, conn, params={"partido": sigla_partido})
