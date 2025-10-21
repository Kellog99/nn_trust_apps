# Mock server

1) make mock_server the current directory
2) RUN  `python app.py`

commands to try api's:\

curl -X GET "http://127.0.0.1:8000/report/getResult?id=12345" \
     -H "accept: application/json"

curl -X GET "http://127.0.0.1:8000/benchmark/getResult?dataset=my_dataset&task=my_task" \
     -H "accept: application/json"

curl -X GET "http://127.0.0.1:8000/attacks/getInfo" \
     -H "accept: application/json"


