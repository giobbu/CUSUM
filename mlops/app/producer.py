from kafka import KafkaProducer
import json
import time
from loguru import logger
import time
import numpy as np

producer = KafkaProducer(
    bootstrap_servers='broker:29092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

while True:
    i = np.random.randint(0, 100)
    message = {"number": i,\
               "sent_timestamp": time.time()}
    producer.send('test-topic', message)
    logger.info(f"Sent: {message}")
    time.sleep(1)


producer.flush()