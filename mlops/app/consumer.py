import json
from loguru import logger
import time
from kafka import KafkaConsumer
from prometheus_client import start_http_server, Gauge
from setting.consumer import  KafkaConsumerSettings
from setting.logger import LoggerSettings
from detector.cusum import PHTest_Detector

# -------------------------
# Configure Loguru Logging
# -------------------------
logger_settings = LoggerSettings()
logger.remove()  # remove default stderr handler
logger.add(
    logger_settings.CONSUMER_FILE_PATH,  # log file path
    level=logger_settings.LEVEL,  # log level
    rotation=logger_settings.ROTATION,  # rotate
    retention=logger_settings.RETENTION,  # keep logs for 10 days
    compression=logger_settings.COMPRESSION,  # compress old logs
    enqueue=logger_settings.ENQUEUE,  # use multiprocessing queue
    backtrace=logger_settings.BACKTRACE,  # include backtrace in logs
    diagnose=logger_settings.DIAGNOSE,  # include variable values in logs
    format=logger_settings.FORMAT  # log message format
)

# Optional: still log to console
logger.add(
    lambda msg: print(msg, end=""),
    level=logger_settings.LEVEL,
)


# -------------------------
# Configure Kafka Consumer
# -------------------------
consumer_settings = KafkaConsumerSettings()
consumer = KafkaConsumer(
    consumer_settings.TOPIC,
    bootstrap_servers=consumer_settings.BOOTSTRAP_SERVERS,
    auto_offset_reset=consumer_settings.AUTO_OFFSET_RESET,
    group_id=consumer_settings.GROUP_ID,
    value_deserializer=lambda m: json.loads(m.decode(consumer_settings.UTF8_DECODER)
                                            )
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
        
        logger.info(f"\n"
            f"Value: {observation}, \n"
            f"Sent at: {sent_timestamp}, \n"
            f"Received at: {time.time()}, \n"
            f"Delay: {get_delay:.2f} seconds"
        )

        logger.info(f"\n"
            f"PH-Test Detector Results: \n"
            f"Positive Change: {pos[0]}, \n"
            f"Negative Change: {neg[0]}, \n"
            f"Change Detected: {is_change}, \n"
            f"Metadata: \n --- {ph_test_detector}"
        )
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