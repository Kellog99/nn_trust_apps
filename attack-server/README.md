To run the celery:
`celery -A celery_worker.celery worker` starts celery worker (consumer) process

To run celery with flower:
`celery -A celery_worker.celery flower` starts flower for celery monitoring
Then execute a request to the required route that implements a celery task.

In the data-quality_gui public folder two default folders must exist:
`public/titann/datasets`
`public/titann/models`

If 'database.db' doesn't exist in the root directory of the project(current directory must be nn_trust_apps), run the following command to create the db (SQLite)
`python attack-server/database/create_sql_lite_db.py`

To run the backend (current directory must be nn_trust_apps):
`python attack-server/app.py`

TO run the front-end (current directory must be attack-server/submodules/data-quality_gui), run the following command in another
terminal:
`npm install #only for the first time` 
`npm run dev` 



