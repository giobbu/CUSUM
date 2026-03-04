from pydantic import BaseModel

class LoggerSettings(BaseModel):
    CONSUMER_FILE_PATH: str = "logs/consumer.log"
    PRODUCER_FILE_PATH: str = "logs/producer.log"
    LEVEL: str = "INFO"
    ROTATION: str = "10 MB"
    RETENTION: str = "10 days"
    COMPRESSION: str = "zip"
    ENQUEUE: bool = True
    BACKTRACE: bool = True
    DIAGNOSE: bool = True
    FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
