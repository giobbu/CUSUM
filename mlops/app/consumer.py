from kafka import KafkaConsumer
import json
from loguru import logger
from datetime import datetime

consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers='broker:29092',
    auto_offset_reset='earliest',
    group_id='my-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    logger.info(f"Received: {message.value} at {datetime.now()}")