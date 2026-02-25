import json
from loguru import logger
import time
from kafka import KafkaConsumer
from prometheus_client import start_http_server, Gauge

from detector.cusum import PHTest_Detector

consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers='broker:29092',
    auto_offset_reset='earliest',
    group_id='my-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

delay_histogram = Gauge('message_delay', 'Delay in seconds between message sent and received' )
observation_histogram = Gauge('message_observations', 'Observed message number')
alarm_gauge = Gauge('change_detection_alarm', 'Alarm for change detection (1 if change detected, 0 otherwise)')
alarm_pos_increase_gauge = Gauge('alarm_pos_increase', 'Positive increase in CUSUM when change is detected')
alarm_neg_increase_gauge = Gauge('alarm_neg_increase', 'Negative increase in CUSUM when change is detected')

# Start Prometheus metrics server on port 8000
start_http_server(8000)

# Initialize CUSUM detector
ph_test_detector = PHTest_Detector(warmup_period=30, delta=0.5, threshold=2)

try:
    for message in consumer:

        observation = message.value.get("observation")
        sent_timestamp = message.value.get("sent_timestamp")

        pos, neg, is_change = ph_test_detector.detection(observation)

        get_delay = time.time() - sent_timestamp
        logger.info(f"Value: {observation}, Sent at: {sent_timestamp}, Received at: {time.time()}, Delay: {get_delay:.2f} seconds")
        logger.info(f"PHTest Detector - Pos: {pos}, Neg: {neg}, Change Detected: {is_change}")
        logger.info("-" * 50)
        if is_change:
            logger.warning("Change detected in the data stream!")
        logger.info("-" * 50)

        # Update Prometheus metric delay histogram
        delay_histogram.set(get_delay)
        # Update Prometheus metric observation histogram
        observation_histogram.set(observation)
        # Update Prometheus metric alarm gauge
        alarm_gauge.set(1 if is_change else 0)
        # Update Prometheus metric for positive and negative increases in CUSUM
        alarm_pos_increase_gauge.set(pos[0])
        alarm_neg_increase_gauge.set(neg[0])

except KeyboardInterrupt:
    logger.info("Consumer stopped by user.")
finally:
    consumer.close()