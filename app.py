from __future__ import annotations

import os

from flask import Flask
from pymongo import MongoClient

from models import db
from config import build_postgres_url, build_mongo_uri, mongo_db_name
from routes import register_routes


def create_app() -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config["SECRET_KEY"] = "dev-secret"
    os.makedirs(app.instance_path, exist_ok=True)

    app.config["SQLALCHEMY_DATABASE_URI"] = build_postgres_url()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    with app.app_context():
        db.create_all()

    mongo_client = MongoClient(build_mongo_uri())
    mongo_db = mongo_client[mongo_db_name()]
    graphs_coll = mongo_db["graphs"]

    register_routes(app, graphs_coll)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
