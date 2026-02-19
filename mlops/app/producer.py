from kafka import KafkaProducer
import json
import time
from loguru import logger
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers='broker:29092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for i in range(10):
    message = {"number": i}
    producer.send('test-topic', message)
    logger.info(f"Sent: {message} at {datetime.now()}")
    time.sleep(1)

producer.flush()