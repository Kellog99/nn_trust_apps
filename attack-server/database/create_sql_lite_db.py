from sqlmodel import SQLModel, create_engine
import os, json
from models import BenchmarkJob, AttackJob
from pathlib import Path
import logging

project_root =  Path(__file__).parent.parent
database_config_path = project_root / "resources" / "config.json"

with open(database_config_path) as f:
    config = json.load(f)
    sqlite_file_name = config["db_name"]
    sqlite_url = config["db_url"]

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

if __name__ == "__main__":
    if os.path.isfile(project_root / config['db_name']):
        raise FileExistsError(f"A database object already exist in the target location: {project_root / config['db_name']}")
    SQLModel.metadata.create_all(engine)
    print(f"Database object succesfully created at {project_root / config['db_name']}")