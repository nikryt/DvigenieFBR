from sqlalchemy import Column, Integer, String, LargeBinary, BigInteger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import mapped_column, sessionmaker

# Настройка подключения к БД
engine = create_async_engine('sqlite+aiosqlite:///faces.db')
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True)
    tg_id = mapped_column(BigInteger)
    name = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)

async  def async_main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)