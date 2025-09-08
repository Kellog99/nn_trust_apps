from abc import ABC, abstractmethod
import json
from typing import List
import redis

class Extractor(ABC):

    @abstractmethod
    def extract() -> List[str]:
        """
        This functions returns the list of Celery task-ids of Celery tasks running in the result backend.
        
        """
        raise NotImplementedError("This is an abstract method.")
        


class CeleryRedisExtractor(Extractor):

    def __init__(self, 
                 host: str = 'localhost', 
                 port: int = 6379, 
                 db_number : int = 0, 
                 queue_name : str = 'unacked'):
        self.queue_name = queue_name
        self.host = host
        self.port = port
        self.r_client = redis.Redis(host=host, port=port, db=db_number)

    def extract_task_ids(self,data: dict) -> List[str]:
        """
        Extract task IDs from Celery message queue output.
        
        Args:
            data: Either a string representation of the data structure or 
                the actual dictionary containing message data
        
        Returns:
            List of task IDs found in the data
        """
        task_ids = []
        
        for key, value in data.items():
            # Decode bytes to string if needed
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            # Parse JSON
            json_list = json.loads(value)
            
            # Extract the "id" from headers
            task_id = json_list[0]['headers']['id']
            task_ids.append(task_id)
            print(f"Key: {key}, ID: {task_id}")
            
        return task_ids
    
    def extract(self) -> List[str]:
        """
        This functions returns the list of Celery task-ids of Celery tasks running in the result backend.
        
        """
        result = self.r_client.hgetall(self.queue_name)
        return self.extract_task_ids(result)

