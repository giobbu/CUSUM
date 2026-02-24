import json
from loguru import logger
import time
from kafka import KafkaConsumer
from prometheus_client import start_http_server, Gauge


consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers='broker:29092',
    auto_offset_reset='earliest',
    group_id='my-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)


delay_histogram = Gauge('message_delay_seconds', 'Delay in seconds between message sent and received' )

observation_histogram = Gauge('message_observations', 'Observed message number')

# Start Prometheus metrics server on port 8000
start_http_server(8000)


for message in consumer:
    logger.info(f"Received: {message.value} at {time.time()}")

    message_number = message.value.get("number")
    sent_timestamp = message.value.get("sent_timestamp")

    get_delay = time.time() - sent_timestamp
    logger.info(f"Message number: {message_number}, Sent at: {sent_timestamp}, Received at: {time.time()}, Delay: {get_delay:.2f} seconds")

    # Update Prometheus metric delay histogram
    delay_histogram.set(get_delay)

    # Update Prometheus metric observation histogram
    observation_histogram.set(message_number)

    time.sleep(1)