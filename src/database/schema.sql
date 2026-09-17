-- ============================================================
-- PostgreSQL Database Schema
-- AI-Powered Parkinson's Disease Severity Assessment
-- ============================================================
-- Tables:
--   users        — System users (clinicians, admins)
--   patients     — Patient records
--   predictions  — Model prediction results
--   reports      — Generated PDF reports
--   logs         — System audit logs
-- ============================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────
-- USERS TABLE
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username        VARCHAR(50)  UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name       VARCHAR(150),
    role            VARCHAR(20)  NOT NULL DEFAULT 'clinician'
                    CHECK (role IN ('admin', 'clinician', 'researcher', 'viewer')),
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_login      TIMESTAMPTZ
);

CREATE INDEX idx_users_email    ON users (email);
CREATE INDEX idx_users_username ON users (username);

-- ─────────────────────────────────────────────
-- PATIENTS TABLE
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_code    VARCHAR(20)  UNIQUE NOT NULL,  -- e.g. PAT-2024-001
    first_name      VARCHAR(100) NOT NULL,
    last_name       VARCHAR(100) NOT NULL,
    date_of_birth   DATE,
    gender          VARCHAR(10)  CHECK (gender IN ('Male', 'Female', 'Other', 'Prefer not to say')),
    age             SMALLINT     GENERATED ALWAYS AS (
                        EXTRACT(YEAR FROM AGE(date_of_birth))::SMALLINT
                    ) STORED,
    email           VARCHAR(255),
    phone           VARCHAR(20),
    address         TEXT,
    -- Clinical info
    diagnosis_date  DATE,
    disease_duration_years FLOAT,
    medications     TEXT[],                         -- List of current medications
    comorbidities   TEXT[],
    referring_clinician VARCHAR(150),
    notes           TEXT,
    -- Audit
    created_by      UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_patients_code     ON patients (patient_code);
CREATE INDEX idx_patients_name     ON patients (last_name, first_name);
CREATE INDEX idx_patients_created  ON patients (created_at DESC);

-- ─────────────────────────────────────────────
-- PREDICTIONS TABLE
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id          UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    created_by          UUID REFERENCES users(id) ON DELETE SET NULL,
    assessed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Modality availability flags
    voice_available     BOOLEAN NOT NULL DEFAULT FALSE,
    eeg_available       BOOLEAN NOT NULL DEFAULT FALSE,
    gait_available      BOOLEAN NOT NULL DEFAULT FALSE,

    -- Input file references (paths or S3 keys)
    voice_file_path     TEXT,
    eeg_file_path       TEXT,
    gait_file_path      TEXT,

    -- Extracted features (stored as JSONB)
    voice_features      JSONB,
    eeg_features        JSONB,
    gait_features       JSONB,

    -- Model predictions
    updrs_score         FLOAT NOT NULL,            -- Predicted UPDRS (0–108)
    updrs_true          FLOAT,                     -- Ground truth (if available)
    severity_class      SMALLINT NOT NULL           -- 0=Mild, 1=Moderate, 2=Severe
                        CHECK (severity_class IN (0, 1, 2)),
    severity_label      VARCHAR(10) NOT NULL
                        CHECK (severity_label IN ('Mild', 'Moderate', 'Severe')),
    confidence          FLOAT NOT NULL              -- Class confidence (0–1)
                        CHECK (confidence BETWEEN 0 AND 1),

    -- Class probabilities
    prob_mild           FLOAT CHECK (prob_mild BETWEEN 0 AND 1),
    prob_moderate       FLOAT CHECK (prob_moderate BETWEEN 0 AND 1),
    prob_severe         FLOAT CHECK (prob_severe BETWEEN 0 AND 1),

    -- Model metadata
    model_version       VARCHAR(30) NOT NULL DEFAULT 'v1.0',
    model_checkpoint    VARCHAR(255),

    -- Embeddings (compressed or reference)
    fused_embedding     FLOAT[],                   -- Optional: store for t-SNE/analysis

    -- Processing time
    inference_time_ms   FLOAT,
    notes               TEXT,

    -- Constraints
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT prob_sum_check CHECK (
        ABS((COALESCE(prob_mild, 0) + COALESCE(prob_moderate, 0) + COALESCE(prob_severe, 0)) - 1.0) < 0.05
    )
);

CREATE INDEX idx_pred_patient    ON predictions (patient_id, assessed_at DESC);
CREATE INDEX idx_pred_severity   ON predictions (severity_label);
CREATE INDEX idx_pred_updrs      ON predictions (updrs_score);
CREATE INDEX idx_pred_assessed   ON predictions (assessed_at DESC);

-- ─────────────────────────────────────────────
-- REPORTS TABLE
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reports (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id   UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    patient_id      UUID NOT NULL REFERENCES patients(id)    ON DELETE CASCADE,
    generated_by    UUID REFERENCES users(id) ON DELETE SET NULL,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Report metadata
    report_type     VARCHAR(30) NOT NULL DEFAULT 'full'
                    CHECK (report_type IN ('full', 'summary', 'voice_only', 'research')),
    format          VARCHAR(10) NOT NULL DEFAULT 'pdf'
                    CHECK (format IN ('pdf', 'html', 'json')),

    -- Storage
    file_path       TEXT NOT NULL,                  -- Local path or cloud URL
    file_size_kb    INTEGER,

    -- Report content summary
    included_sections TEXT[],                       -- ['voice', 'eeg', 'gait', 'shap', 'recommendations']
    xai_included    BOOLEAN NOT NULL DEFAULT FALSE,
    recommendations TEXT,

    -- Status
    status          VARCHAR(20) NOT NULL DEFAULT 'generated'
                    CHECK (status IN ('generating', 'generated', 'failed', 'archived')),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_reports_patient   ON reports (patient_id, generated_at DESC);
CREATE INDEX idx_reports_pred      ON reports (prediction_id);

-- ─────────────────────────────────────────────
-- AUDIT LOGS TABLE
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS logs (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    action          VARCHAR(100) NOT NULL,          -- 'predict', 'login', 'report_generate', etc.
    resource_type   VARCHAR(50),                    -- 'patient', 'prediction', 'report'
    resource_id     UUID,
    request_ip      VARCHAR(45),
    user_agent      TEXT,
    status_code     SMALLINT,
    response_time_ms FLOAT,
    error_message   TEXT,
    log_metadata    JSONB
);

CREATE INDEX idx_logs_timestamp   ON logs (timestamp DESC);
CREATE INDEX idx_logs_user        ON logs (user_id, timestamp DESC);
CREATE INDEX idx_logs_action      ON logs (action);
CREATE INDEX idx_logs_resource    ON logs (resource_type, resource_id);

-- ─────────────────────────────────────────────
-- VIEWS
-- ─────────────────────────────────────────────

-- Summary view: patient + latest prediction
CREATE OR REPLACE VIEW v_patient_latest_assessment AS
SELECT
    p.id            AS patient_id,
    p.patient_code,
    p.first_name || ' ' || p.last_name AS full_name,
    p.age,
    p.gender,
    pr.id           AS prediction_id,
    pr.assessed_at,
    pr.updrs_score,
    pr.severity_label,
    pr.confidence,
    pr.voice_available,
    pr.eeg_available,
    pr.gait_available,
    pr.model_version
FROM patients p
LEFT JOIN LATERAL (
    SELECT * FROM predictions
    WHERE patient_id = p.id
    ORDER BY assessed_at DESC
    LIMIT 1
) pr ON TRUE;

-- Severity distribution
CREATE OR REPLACE VIEW v_severity_distribution AS
SELECT
    severity_label,
    COUNT(*)                                   AS total,
    ROUND(AVG(updrs_score)::NUMERIC, 2)        AS avg_updrs,
    ROUND(MIN(updrs_score)::NUMERIC, 2)        AS min_updrs,
    ROUND(MAX(updrs_score)::NUMERIC, 2)        AS max_updrs,
    ROUND(AVG(confidence)::NUMERIC, 3)         AS avg_confidence,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()::NUMERIC, 1) AS pct
FROM predictions
GROUP BY severity_label
ORDER BY severity_class;

-- ─────────────────────────────────────────────
-- AUTO-UPDATE TRIGGERS
-- ─────────────────────────────────────────────

-- Trigger function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_patients_updated
    BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ─────────────────────────────────────────────
-- SEED DATA (Admin User)
-- ─────────────────────────────────────────────

-- Default admin user (password: Admin@123 — change in production)
-- hashed_password = bcrypt hash of 'Admin@123'
INSERT INTO users (username, email, hashed_password, full_name, role)
VALUES (
    'admin',
    'admin@parkinsons-ai.local',
    '$2b$12$placeholder_change_in_production_xxxxxxxxxxxxxxxxxxxxxxxxx',
    'System Administrator',
    'admin'
) ON CONFLICT (username) DO NOTHING;
