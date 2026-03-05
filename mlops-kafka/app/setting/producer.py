from pydantic import BaseModel

class KafkaProducerSettings(BaseModel):
    BOOTSTRAP_SERVERS: str = 'broker:29092'
    UTF8_ENCODER: str = 'utf-8'