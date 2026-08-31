# scripts/reset_db.py
import chromadb
import yaml

def reset_chroma():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    client = chromadb.PersistentClient(path=config["chroma"]["persist_directory"])
    collection_name = config["chroma"]["collection_name"]

    try:
        client.delete_collection(name=collection_name)
        print(f"Successfully deleted collection: {collection_name}")
    except Exception as e:
        print(f"Collection does not exist or already cleared: {e}")

if __name__ == "__main__":
    reset_chroma()