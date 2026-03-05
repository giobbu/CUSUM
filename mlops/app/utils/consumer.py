import json
from kafka import KafkaConsumer

def setup_consumer(settings):
    consumer = KafkaConsumer(
        settings.TOPIC,
        bootstrap_servers=settings.BOOTSTRAP_SERVERS,
        auto_offset_reset=settings.AUTO_OFFSET_RESET,
        group_id=settings.GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode(settings.UTF8_DECODER)
                                                )
        )
    return consumer