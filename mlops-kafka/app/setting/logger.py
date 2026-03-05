from pydantic import BaseModel

class ProducerLoggerSettings(BaseModel):
    SERVICE: str = "kafka-producer"
    FILE_PATH: str = "logs/producer.jsonl"
    LEVEL: str = "INFO"

class ConsumerLoggerSettings(BaseModel):
    SERVICE: str = "kafka-consumer"
    FILE_PATH: str = "logs/consumer.jsonl"
    LEVEL: str = "INFO"


