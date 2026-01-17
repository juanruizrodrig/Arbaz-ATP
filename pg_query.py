#!/usr/bin/env python3
import argparse
import csv
import sys
from typing import Optional

import psycopg
from psycopg.rows import dict_row


def run_query(
    host: str,
    port: int,
    dbname: str,
    user: str,
    password: str,
    sql: str,
    params: Optional[list[str]] = None,
    statement_timeout_ms: int = 30000,
    csv_path: Optional[str] = None,
    max_rows: int = 200,
) -> int:
    dsn = f"host={host} port={port} dbname={dbname} user={user} password={password}"

    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            # Evita consultas colgadas
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {statement_timeout_ms};")

                # Ejecuta la query
                cur.execute(sql, params or None)

                # Si no devuelve filas (INSERT/UPDATE/DDL)
                if cur.description is None:
                    conn.commit()
                    print(f"OK. Filas afectadas: {cur.rowcount}")
                    return 0

                rows = cur.fetchmany(max_rows)
                if not rows:
                    print("(0 filas)")
                    return 0

                headers = list(rows[0].keys())

                # Exportar a CSV si se pide
                if csv_path:
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=headers)
                        w.writeheader()
                        w.writerows(rows)
                    print(f"Exportado a CSV: {csv_path} (primeras {len(rows)} filas)")
                    return 0

                # Imprimir como tabla simple
                col_widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}

                def fmt_row(d):
                    return " | ".join(str(d.get(h, "")).ljust(col_widths[h]) for h in headers)

                print(fmt_row({h: h for h in headers}))
                print("-+-".join("-" * col_widths[h] for h in headers))
                for r in rows:
                    print(fmt_row(r))

                # Aviso si hay más filas
                more = cur.fetchmany(1)
                if more:
                    print(f"\n(Mostradas {len(rows)} filas. Hay más; usa --max-rows o añade LIMIT.)")

                return 0

    except psycopg.errors.QueryCanceled as e:
        print(f"ERROR: Query cancelada por timeout ({statement_timeout_ms} ms).", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description="Ejecuta consultas SQL en PostgreSQL sin psql interactivo.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db", default="mi_bd")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--sql", help="Consulta SQL (entre comillas).")
    parser.add_argument("--file", help="Ruta a archivo .sql para ejecutar.")
    parser.add_argument("--param", action="append", help="Parámetros posicionales ($1, $2...) (puedes repetir).")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="statement_timeout en ms.")
    parser.add_argument("--csv", help="Ruta de salida CSV (exporta primeras --max-rows filas).")
    parser.add_argument("--max-rows", type=int, default=200, help="Máximo de filas a mostrar/exportar.")
    args = parser.parse_args()

    if not args.sql and not args.file:
        parser.error("Debes indicar --sql o --file")

    sql = args.sql
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            sql = f.read()

    # Nota: psycopg3 usa %s para parámetros. Si quieres $1, $2, también funciona en PostgreSQL,
    # pero en psycopg es más típico usar %s, %s...
    # Este script pasa params como lista para placeholders %s.
    params = args.param or None

    exit_code = run_query(
        host=args.host,
        port=args.port,
        dbname=args.db,
        user=args.user,
        password=args.password,
        sql=sql,
        params=params,
        statement_timeout_ms=args.timeout_ms,
        csv_path=args.csv,
        max_rows=args.max_rows,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
