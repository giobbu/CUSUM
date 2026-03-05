from kafka import KafkaProducer
import json

def setup_producer(settings):
    producer = KafkaProducer(
                            bootstrap_servers=settings.BOOTSTRAP_SERVERS,
                            value_serializer=lambda v: json.dumps(v).encode(settings.UTF8_ENCODER)
                        )
    return producer