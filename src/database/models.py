"""
============================================================
SQLAlchemy ORM Models
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Async SQLAlchemy 2.0 models for all database tables.
Supports both sync and async operations.
============================================================
"""

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Column, DateTime,
    Enum, Float, ForeignKey, Index, Integer, SmallInteger,
    String, Text, ARRAY,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


# ─────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class UserRole(str, enum.Enum):
    admin      = "admin"
    clinician  = "clinician"
    researcher = "researcher"
    viewer     = "viewer"


class SeverityLabel(str, enum.Enum):
    Mild     = "Mild"
    Moderate = "Moderate"
    Severe   = "Severe"


class ReportType(str, enum.Enum):
    full       = "full"
    summary    = "summary"
    voice_only = "voice_only"
    research   = "research"


class ReportStatus(str, enum.Enum):
    generating = "generating"
    generated  = "generated"
    failed     = "failed"
    archived   = "archived"


# ─────────────────────────────────────────────
# User Model
# ─────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username        = Column(String(50),  unique=True, nullable=False)
    email           = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name       = Column(String(150))
    role            = Column(Enum(UserRole), nullable=False, default=UserRole.clinician)
    is_active       = Column(Boolean, nullable=False, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login      = Column(DateTime(timezone=True))

    # Relationships
    patients    = relationship("Patient",    back_populates="created_by_user")
    predictions = relationship("Prediction", back_populates="created_by_user")
    reports     = relationship("Report",     back_populates="generated_by_user")
    logs        = relationship("AuditLog",   back_populates="user")

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# ─────────────────────────────────────────────
# Patient Model
# ─────────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"

    id                    = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_code          = Column(String(20), unique=True, nullable=False)
    first_name            = Column(String(100), nullable=False)
    last_name             = Column(String(100), nullable=False)
    date_of_birth         = Column(DateTime)
    gender                = Column(String(20))
    email                 = Column(String(255))
    phone                 = Column(String(20))
    address               = Column(Text)
    diagnosis_date        = Column(DateTime)
    disease_duration_years= Column(Float)
    medications           = Column(ARRAY(Text))
    comorbidities         = Column(ARRAY(Text))
    referring_clinician   = Column(String(150))
    notes                 = Column(Text)
    created_by            = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at            = Column(DateTime(timezone=True), server_default=func.now())
    updated_at            = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    created_by_user = relationship("User",       back_populates="patients",    foreign_keys=[created_by])
    predictions     = relationship("Prediction", back_populates="patient",     cascade="all, delete-orphan")
    reports         = relationship("Report",     back_populates="patient",     cascade="all, delete-orphan")

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def __repr__(self):
        return f"<Patient {self.patient_code}: {self.full_name}>"


# ─────────────────────────────────────────────
# Prediction Model
# ─────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id                  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id          = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    created_by          = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    assessed_at         = Column(DateTime(timezone=True), server_default=func.now())

    # Modality flags
    voice_available     = Column(Boolean, nullable=False, default=False)
    eeg_available       = Column(Boolean, nullable=False, default=False)
    gait_available      = Column(Boolean, nullable=False, default=False)

    # File paths
    voice_file_path     = Column(Text)
    eeg_file_path       = Column(Text)
    gait_file_path      = Column(Text)

    # Features (JSONB)
    voice_features      = Column(JSONB)
    eeg_features        = Column(JSONB)
    gait_features       = Column(JSONB)

    # Predictions
    updrs_score         = Column(Float, nullable=False)
    updrs_true          = Column(Float)
    severity_class      = Column(SmallInteger, nullable=False)
    severity_label      = Column(Enum(SeverityLabel), nullable=False)
    confidence          = Column(Float, nullable=False)

    # Class probabilities
    prob_mild           = Column(Float)
    prob_moderate       = Column(Float)
    prob_severe         = Column(Float)

    # Model metadata
    model_version       = Column(String(30), nullable=False, default="v1.0")
    model_checkpoint    = Column(String(255))

    # Performance
    inference_time_ms   = Column(Float)
    notes               = Column(Text)
    created_at          = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    patient          = relationship("Patient",   back_populates="predictions")
    created_by_user  = relationship("User",      back_populates="predictions", foreign_keys=[created_by])
    reports          = relationship("Report",    back_populates="prediction",  cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Prediction {self.id} | UPDRS={self.updrs_score:.1f} | {self.severity_label}>"


# ─────────────────────────────────────────────
# Report Model
# ─────────────────────────────────────────────

class Report(Base):
    __tablename__ = "reports"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id    = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False)
    patient_id       = Column(UUID(as_uuid=True), ForeignKey("patients.id",    ondelete="CASCADE"), nullable=False)
    generated_by     = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    generated_at     = Column(DateTime(timezone=True), server_default=func.now())

    report_type      = Column(Enum(ReportType), nullable=False, default=ReportType.full)
    format           = Column(String(10), nullable=False, default="pdf")
    file_path        = Column(Text, nullable=False)
    file_size_kb     = Column(Integer)
    included_sections= Column(ARRAY(Text))
    xai_included     = Column(Boolean, nullable=False, default=False)
    recommendations  = Column(Text)
    status           = Column(Enum(ReportStatus), nullable=False, default=ReportStatus.generated)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    prediction       = relationship("Prediction", back_populates="reports")
    patient          = relationship("Patient",    back_populates="reports")
    generated_by_user= relationship("User",      back_populates="reports", foreign_keys=[generated_by])


# ─────────────────────────────────────────────
# Audit Log Model
# ─────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "logs"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp        = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    user_id          = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action           = Column(String(100), nullable=False)
    resource_type    = Column(String(50))
    resource_id      = Column(UUID(as_uuid=True))
    request_ip       = Column(String(45))
    user_agent       = Column(Text)
    status_code      = Column(SmallInteger)
    response_time_ms = Column(Float)
    error_message    = Column(Text)
    log_metadata     = Column(JSONB)

    user = relationship("User", back_populates="logs")


# ─────────────────────────────────────────────
# Database Session Factory
# ─────────────────────────────────────────────

from sqlalchemy.ext.asyncio import async_sessionmaker
from contextlib import asynccontextmanager

_engine = None
_session_factory = None


def init_engine(database_url: str):
    """Initialize async SQLAlchemy engine."""
    global _engine, _session_factory
    _engine = create_async_engine(
        database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


@asynccontextmanager
async def get_session():
    """Async context manager for database sessions."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_engine() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all_tables():
    """Create all tables from ORM models."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[DB] All tables created successfully")
