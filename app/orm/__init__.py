import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")


class Base(DeclarativeBase):
    pass


class Transcription(Base):
    __tablename__ = "transcriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_email: Mapped[str] = mapped_column(String, nullable=False)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    transcript: Mapped[str] = mapped_column(String, nullable=False)
    formatted_transcript: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_email: Mapped[str] = mapped_column(String, nullable=False)
    memory: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )


def init_db():
    # only create table if they did not exist
    engine = create_engine(SUPABASE_DB_URL)
    Base.metadata.create_all(engine)


def delete_db():
    engine = create_engine(SUPABASE_DB_URL)
    Base.metadata.drop_all(engine)
