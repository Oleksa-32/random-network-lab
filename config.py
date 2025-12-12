from __future__ import annotations

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()

ALLOWED_EDGE_EXT = {".txt", ".csv", ".edgelist"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POPULAR_DIR = os.path.join(BASE_DIR, "data", "popular")

POPULAR_NETWORKS = [
    {
        "slug": "Number of nodes 1005",
        "filename": "email-Eu-core.txt",
        "title": "Email communication links between members of the institution",
        "description": "This network represents the core of the email-EuAll network, which also contains links between members of the institution and people outside of the institution (although the node IDs are not the same).",
    },
    {
        "slug": "wiki-Vote",
        "filename": "wiki-Vote.txt",
        "title": "Wikipedia ",
        "description": "Wikipedia adminship vote network till January 2008",
    },
    {
        "slug": "congress edgelist",
        "filename": "congress.edgelist",
        "title": "Twitter interaction network  ",
        "description": "This network represents the Twitter interaction network for the 117th United States Congress, both House of Representatives and Senate. The base data was collected via the Twitter’s API, then the empirical transmission probabilities were quantified according to the fraction of times one member retweeted, quote tweeted, replied to, or mentioned another member’s tweet. See the publication for more details.",
    },
]

DEFAULT_METRICS = [
    "avg_clustering",
    "transitivity",
    "avg_shortest_path_length",
    "diameter",
    "assortativity_degree",
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
]


def build_postgres_url() -> str:
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "netlab")
    user = os.getenv("POSTGRES_USER", "netlab")
    password = os.getenv("POSTGRES_PASSWORD", "netlab")
    safe_password = quote_plus(password)
    return f"postgresql+psycopg2://{user}:{safe_password}@{host}:{port}/{db_name}"


def build_mongo_uri() -> str:
    explicit = os.getenv("MONGODB_URI")
    if explicit:
        return explicit
    host = os.getenv("MONGODB_HOST", "localhost")
    port = os.getenv("MONGODB_PORT", "27017")
    return f"mongodb://{host}:{port}"


def mongo_db_name() -> str:
    return os.getenv("MONGODB_DB", "random_netlab")
