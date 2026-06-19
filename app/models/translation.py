from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Translation(Base):
    __tablename__ = "translations"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    input_type      = Column(String, nullable=False)   # "alphabet" or "word"
    detected_signs  = Column(Text, nullable=False)     # raw signs e.g. "H,E,L,L,O"
    result_text     = Column(Text, nullable=False)     # final sentence e.g. "HELLO"
    confidence      = Column(Float, nullable=True)     # avg confidence
    duration_ms     = Column(Integer, nullable=True)   # how long the session lasted
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="translations")