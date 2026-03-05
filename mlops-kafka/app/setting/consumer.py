from pydantic import BaseModel

class KafkaConsumerSettings(BaseModel):
    TOPIC: str = 'test-topic'
    BOOTSTRAP_SERVERS: str = 'broker:29092'
    AUTO_OFFSET_RESET: str = 'earliest'
    GROUP_ID: str = 'my-group'
    UTF8_DECODER: str = 'utf-8'
    

