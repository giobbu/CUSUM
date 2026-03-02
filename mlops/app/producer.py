from kafka import KafkaProducer
import json
import time
from loguru import logger
import time
import numpy as np
from setting.producer import KafkaProducerSettings
from utils.data import generate_mean_and_std_dev_break_point, plot_observations_with_breaks

producer_settings = KafkaProducerSettings()

producer = KafkaProducer(
                        bootstrap_servers=producer_settings.BOOTSTRAP_SERVERS,
                        value_serializer=lambda v: json.dumps(v).encode(producer_settings.UTF8_ENCODER)
                    )

count = 0
mean, std_dev, next_break_point = generate_mean_and_std_dev_break_point()  # Initial mean and std_dev
list_observations = []
list_mean = [mean]
list_breaks = [next_break_point]

for _ in range(5000):  # Log the initial mean and std_dev for the first 10 iterations
# while True:
    if count % next_break_point == 0:
        mean, std_dev, break_point = generate_mean_and_std_dev_break_point()
        next_break_point += break_point  # Set the next breakpoint
        logger.info(f"Updated mean to {mean:.2f}, std_dev to {std_dev:.2f}, current break point: {count}, next break point: {next_break_point}")
        list_mean.append(mean)
        list_breaks.append(next_break_point)
    obs = np.random.normal(loc=mean, scale=std_dev)  # Generate a random number from the normal distribution
    message = {"observation": obs, "sent_timestamp": time.time(), "count": count}
    producer.send('test-topic', message)
    logger.info(f"Sent: {message}")
    time.sleep(5)
    count += 1
    list_observations.append(obs)
    
producer.flush()
logger.info("Break points: " + ", ".join(str(bp) for bp in list_breaks))

plot_observations_with_breaks(list_observations, list_breaks)


