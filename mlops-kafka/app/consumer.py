from loguru import logger
import time
from prometheus_client import start_http_server, Gauge
from setting.consumer import  KafkaConsumerSettings
from setting.logger import ConsumerLoggerSettings
from detector.cusum import PHTest_Detector

from utils.consumer import setup_consumer
from utils.logger import setup_logger

logger_settings = ConsumerLoggerSettings()
logger = setup_logger(logger_settings)


consumer_settings = KafkaConsumerSettings()
consumer = setup_consumer(consumer_settings)
logger.bind(event="consumer_started",
            bootstrap_servers=consumer_settings.BOOTSTRAP_SERVERS,
            topic=consumer_settings.TOPIC
            ).info("Kafka consumer initialized")


delay_histogram = Gauge('message_delay', 'Delay in seconds between message sent and received' )
observation_histogram = Gauge('message_observations', 'Observed message number')
alarm_gauge = Gauge('change_detection_alarm', 'Alarm for change detection (1 if change detected, 0 otherwise)')
alarm_pos_increase_gauge = Gauge('alarm_pos_increase', 'Positive increase in CUSUM when change is detected')
alarm_neg_increase_gauge = Gauge('alarm_neg_increase', 'Negative increase in CUSUM when change is detected')

# Start Prometheus metrics server on port 8000
PORT = 8000
start_http_server(PORT)
logger.bind(event="prometheus_server_started", 
            port=PORT).info("Prometheus metrics server started")

# Initialize CUSUM detector
ph_test_detector = PHTest_Detector(warmup_period=30, delta=0.5, threshold=2)

try:
    for message in consumer:

        observation = message.value.get("observation")
        sent_timestamp = message.value.get("sent_timestamp")
        pos, neg, is_change = ph_test_detector.detection(observation)

        get_delay = time.time() - sent_timestamp

        logger.bind(
            event="message_consumed",
            observation=observation,
            sent_timestamp=sent_timestamp,
            received_timestamp=time.time(),
            delay=get_delay
            ).info("message consumed and processed")
        
        logger.bind(
            event="change_detection_result",
            observation = float(observation),
            positive_increase =  int(pos[0]),
            negative_increase = int(neg[0]),
            is_change = bool(is_change),
            detector_params={"warmup_period": ph_test_detector.warmup_period,
                      "delta": ph_test_detector.delta,
                      "threshold": ph_test_detector.threshold}
        ).info("CUSUM change detection result")

        if is_change:
            logger.bind(
                event="change_detected",
                observation=observation,
                pos_increase=pos[0],
                neg_increase=neg[0],
                metadata=str(ph_test_detector)
            ).warning("Change detected in the data stream")

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