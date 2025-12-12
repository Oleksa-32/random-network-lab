from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Study(db.Model):
    __tablename__ = "studies"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # "single" або "group"
    mode = db.Column(db.String(16), nullable=False)

    # "er" | "ws" | "ba" | "file" | "popular"
    generator = db.Column(db.String(16), nullable=False)

    # параметри генератора (dict)
    params = db.Column(db.JSON, nullable=False)

    # список метрик, які рахували
    metrics = db.Column(db.JSON, nullable=False)

    # результати:
    # - single: dict {metric_name: value_or_summary}
    # - group:  dict {metric_name: {"mean": ..., "std": ..., "samples": n}}
    results = db.Column(db.JSON, nullable=False)
