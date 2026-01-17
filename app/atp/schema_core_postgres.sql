-- ============================================================
-- Target schema for "Avisos" migration (designed from Tablas.xlsx)
-- Compatible with Azure Database for PostgreSQL (Flexible Server)
-- ============================================================

-- Enable UUID generation (preferred)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create a dedicated schema (optional). If you prefer public, tell me and I'll adjust.
CREATE SCHEMA IF NOT EXISTS core;

-- ---------------------------
-- 1) Reference / Master data
-- ---------------------------

CREATE TABLE IF NOT EXISTS core.insurer (
  insurer_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name            text NOT NULL UNIQUE,
  default_fee_eur numeric(10,2) NULL,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS core.status (
  status_id   smallserial PRIMARY KEY,
  name        text NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS core.responsible (
  responsible_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  display_name   text NOT NULL UNIQUE,
  created_at     timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------
-- 2) Parties / policy context
-- ---------------------------

CREATE TABLE IF NOT EXISTS core.address (
  address_id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  line1        text NOT NULL,
  postal_code  text NULL,
  city         text NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  -- Basic de-dup helper (not a strict constraint)
  normalized_key text GENERATED ALWAYS AS (
    lower(trim(coalesce(line1,''))) || '|' ||
    lower(trim(coalesce(postal_code,''))) || '|' ||
    lower(trim(coalesce(city,'')))
  ) STORED
);

CREATE INDEX IF NOT EXISTS ix_address_normalized_key ON core.address(normalized_key);

CREATE TABLE IF NOT EXISTS core.insured_person (
  insured_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  first_name      text NOT NULL,
  last_name       text NULL,
  phones_raw      text NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  normalized_key  text GENERATED ALWAYS AS (
    lower(trim(coalesce(first_name,''))) || '|' ||
    lower(trim(coalesce(last_name,''))) || '|' ||
    regexp_replace(coalesce(phones_raw,''), '\D', '', 'g')
  ) STORED
);

CREATE INDEX IF NOT EXISTS ix_insured_person_normalized_key ON core.insured_person(normalized_key);

CREATE TABLE IF NOT EXISTS core.policy (
  policy_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  insurer_id    uuid NOT NULL REFERENCES core.insurer(insurer_id),
  policy_number text NOT NULL,
  insured_id    uuid NULL REFERENCES core.insured_person(insured_id),
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (insurer_id, policy_number)
);

-- ---------------------------
-- 3) Main business entity: Aviso / Expediente
-- ---------------------------

CREATE TABLE IF NOT EXISTS core.aviso (
  aviso_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Business key seen in Excel: "2021 / 2025" => order_number=2021, order_year=2025
  order_number    integer NULL,
  order_year      integer NULL,

  aviso_number    text NULL,     -- "Nº Aviso"
  expediente      text NULL,     -- "Expediente"

  insurer_id      uuid NOT NULL REFERENCES core.insurer(insurer_id),
  status_id       smallint NOT NULL REFERENCES core.status(status_id),
  responsible_id  uuid NULL REFERENCES core.responsible(responsible_id),

  notice_date     date NULL,     -- "Fecha de aviso"
  loss_date       date NULL,     -- "Fecha de siniestro"
  effective_date  date NULL,     -- "Fecha de efecto"

  policy_id       uuid NULL REFERENCES core.policy(policy_id),
  insured_id      uuid NULL REFERENCES core.insured_person(insured_id),
  address_id      uuid NULL REFERENCES core.address(address_id),

  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT ck_order_year_reasonable CHECK (order_year IS NULL OR (order_year BETWEEN 1990 AND 2100))
);

-- If order_number/year exist, enforce uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS ux_aviso_order_key
  ON core.aviso(order_year, order_number)
  WHERE order_year IS NOT NULL AND order_number IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_aviso_insurer ON core.aviso(insurer_id);
CREATE INDEX IF NOT EXISTS ix_aviso_status ON core.aviso(status_id);
CREATE INDEX IF NOT EXISTS ix_aviso_responsible ON core.aviso(responsible_id);
CREATE INDEX IF NOT EXISTS ix_aviso_notice_date ON core.aviso(notice_date);

-- ---------------------------
-- 4) Staging table (raw landing)
-- ---------------------------
CREATE SCHEMA IF NOT EXISTS stg;

CREATE TABLE IF NOT EXISTS stg.avisos_raw (
  raw_id              bigserial PRIMARY KEY,
  source_row_number   integer NULL,
  id_text             text NULL,   -- "id" column from Excel
  aviso_number        text NULL,   -- "Nº Aviso"
  expediente          text NULL,
  aseguradora         text NULL,
  fecha_de_aviso      text NULL,
  fecha_de_siniestro  text NULL,
  fecha_de_efecto     text NULL,
  nombre_asegurado    text NULL,
  apellidos           text NULL,
  direccion           text NULL,
  cp                  text NULL,
  ciudad              text NULL,
  telefonos           text NULL,
  poliza              text NULL,
  estado              text NULL,
  responsable         text NULL,
  load_ts             timestamptz NOT NULL DEFAULT now()
);
