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

def generate_mean_and_std_dev_break_point():
    mean = np.random.uniform(-10, 10)  # Random mean between -10 and 10
    std_dev = np.random.uniform(0.1, 1)  # Random standard deviation between 0.1 and 5
    break_point = np.random.randint(50, 100)  # Random break point between 10 and 50 iterations
    return mean, std_dev, break_point

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


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(list_observations)
plt.title('Observations Over Time')
plt.xlabel('Iteration')
plt.ylabel('Observation Value')
for break_point in list_breaks:
    plt.axvline(x=break_point, color='r', linestyle='--', label='Break Points' if break_point == list_breaks[0] else "")
plt.legend()
plt.savefig('observations_plot.png')
plt.show()