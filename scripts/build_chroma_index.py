"""
build_chroma_index.py

(Re)builds the Chroma vector store from every ticket currently in
data/ticket_knowledge_base.db. Run this once after generating the SQLite
KB, and again any time you want to fully re-embed (e.g. after switching
embedding models). Incremental additions from the live pipeline (auto /
admin resolved tickets) don't need this -- they're upserted one at a time
by rag_pipeline.py as they happen.

Usage:
    python scripts/build_chroma_index.py
    python scripts/build_chroma_index.py --config config/config.yaml
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src import db
from src import chroma_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    db.init_db(config["database"]["sqlite_path"])

    tickets = db.fetch_all_tickets(config["database"]["sqlite_path"])
    print(f"Loaded {len(tickets)} tickets from SQLite.")

    collection = chroma_store.get_collection(config)

    batch_size = config["chroma"]["index_batch_size"]
    start = time.time()
    n = chroma_store.index_tickets(collection, tickets, batch_size=batch_size)
    elapsed = time.time() - start

    print(f"Indexed {n} tickets into Chroma collection "
          f"'{config['chroma']['collection_name']}' in {elapsed:.1f}s.")
    print(f"Chroma persisted at: {config['chroma']['persist_directory']}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
