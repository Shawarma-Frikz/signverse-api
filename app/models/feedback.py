from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class PredictionFeedback(Base):
    __tablename__ = "prediction_feedback"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    model_type      = Column(String, nullable=False)   # "alphabet" or "word"
    predicted_label = Column(String, nullable=False)   # what the model said
    correct_label   = Column(String, nullable=True)    # what it actually was (optional)
    confidence      = Column(Float, nullable=True)     # model's confidence
    landmarks       = Column(String, nullable=True)    # JSON string of landmark values
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="feedbacks")