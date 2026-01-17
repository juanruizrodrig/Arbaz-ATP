from dataclasses import dataclass
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor


@dataclass(frozen=True)
class AdminDbConfig:
    host: str = "127.0.0.1"
    port: int = 5432
    dbname: str = "peritaciones"
    user: str = "postgres"
    password_file: Path = Path(r"C:\Users\SG36073\postgresql\pw.txt")
    password_encoding: str = "cp1252"

    def password(self) -> str:
        pw = (
            self.password_file.read_text(encoding=self.password_encoding)
            .replace("\ufeff", "")
            .strip()
        )
        if not pw:
            raise RuntimeError(f"Password vacío en: {self.password_file}")
        return pw


def apply_n8n_schema_perms(target_role: str = "app_user", schema: str = "public") -> None:
    cfg = AdminDbConfig()
    conn = psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password(),
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        print(f"➡️  Aplicando permisos para {target_role} en BD={cfg.dbname}, schema={schema}")

        # Permisos clave para migraciones
        cur.execute(f'GRANT USAGE ON SCHEMA "{schema}" TO "{target_role}";')
        cur.execute(f'GRANT CREATE ON SCHEMA "{schema}" TO "{target_role}";')

        # Permisos útiles (evita errores raros con secuencias/tablas)
        cur.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA "{schema}" TO "{target_role}";')
        cur.execute(f'GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA "{schema}" TO "{target_role}";')

        # Default privileges (para objetos futuros creados por el owner actual)
        cur.execute(f'''
            ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}"
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{target_role}";
        ''')
        cur.execute(f'''
            ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}"
            GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO "{target_role}";
        ''')

        # Verificación
        cur.execute("""
            SELECT
              has_schema_privilege(%s, %s, 'USAGE')  AS has_usage,
              has_schema_privilege(%s, %s, 'CREATE') AS has_create;
        """, [target_role, schema, target_role, schema])
        perms = cur.fetchone()

    conn.close()

    print("✅ Verificación permisos schema:")
    print(f"   USAGE : {perms['has_usage']}")
    print(f"   CREATE: {perms['has_create']}")
    if not perms["has_create"]:
        raise RuntimeError("❌ CREATE sigue siendo FALSE en el schema. Revisa owner/permisos/rol.")


if __name__ == "__main__":
    apply_n8n_schema_perms(target_role="app_user", schema="public")
    print("\n➡️  Ahora relanza n8n. Debería poder ejecutar migraciones.")
