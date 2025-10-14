import os
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./buildabot.db")

# Debug: Print database URL (without sensitive info)
if DATABASE_URL.startswith("postgresql://"):
    print(f"🐘 Using PostgreSQL database")
else:
    print(f"📁 Using SQLite database: {DATABASE_URL}")
    print("⚠️  To use PostgreSQL, set DATABASE_URL in your .env file")


class Base(DeclarativeBase):
    pass


class AgentORM(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(String(1000), nullable=False)
    system_prompt: Mapped[str] = mapped_column(String(8000), nullable=False)
    model: Mapped[str] = mapped_column(String(100), default="gpt-3.5-turbo")
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[int] = mapped_column(Integer, default=1000)
    pipeline_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    workflow_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


# Engine and session setup with improved connection handling
engine_kwargs = {
    "echo": False,
    "future": True,
    "pool_pre_ping": True,  # Verify connections before use
    "pool_recycle": 300,    # Recycle connections every 5 minutes
    "pool_timeout": 20,     # Timeout for getting connection from pool
    "max_overflow": 10,     # Additional connections beyond pool_size
    "pool_size": 5,         # Base number of connections to maintain
}

# Add PostgreSQL-specific settings if using PostgreSQL
if DATABASE_URL.startswith("postgresql://"):
    engine_kwargs.update({
        "connect_args": {
            "connect_timeout": 10,
            "application_name": "buildabot",
            "options": "-c statement_timeout=30000"  # 30 second statement timeout
        }
    })

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


def init_db() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Failed to create database tables: {e}")
        raise


def test_db_connection() -> bool:
    """Test database connection and return True if successful"""
    try:
        with get_session() as session:
            session.execute("SELECT 1")
            print("✅ Database connection test successful")
            return True
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = None
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            session = SessionLocal()
            yield session
            session.commit()
            break
        except Exception as e:
            if session:
                session.rollback()
            
            retry_count += 1
            if retry_count >= max_retries:
                print(f"❌ Database connection failed after {max_retries} retries: {e}")
                raise
            
            print(f"⚠️  Database connection attempt {retry_count} failed: {e}")
            print(f"🔄 Retrying in 1 second...")
            
            # Wait before retry
            import time
            time.sleep(1)
            
            # Force engine to dispose and recreate connections
            if "connection" in str(e).lower() or "abort" in str(e).lower():
                try:
                    engine.dispose()
                except:
                    pass
        finally:
            if session:
                session.close()


