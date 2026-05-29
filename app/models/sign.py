from sqlalchemy import Column, Integer, String
from app.core.database import Base

class Sign(Base):
    __tablename__ = "signs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    category = Column(String, nullable=False)
    description = Column(String, nullable=True)
    video_url = Column(String, nullable=True)