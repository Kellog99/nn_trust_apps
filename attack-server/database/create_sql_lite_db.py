from sqlmodel import SQLModel, create_engine
import os, json
from models import Job

with open(os.path.join("attack-server","resources","config.json")) as f:
    config = json.load(f)
    sqlite_file_name = config["db_name"]
    sqlite_url = config["db_url"]

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)