To run the full configuration

`fastapi dev main.py`
`celery -A celery_worker.celery worker` starts celery worker (consumer) process
`celery -A celery_worker.celery flower` starts flower for celery monitoring
Then execute a request to the required route that implements a celery task.


```
curl --location 'http://127.0.0.1:8000/sum' \
--header 'Content-Type: application/json' \
--data '{
    "x":2,
    "y":3
}'
```