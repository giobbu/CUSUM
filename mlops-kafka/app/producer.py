import time
from loguru import logger
import time
from setting.producer import KafkaProducerSettings
from setting.logger import ProducerLoggerSettings
from utils.data import generate_observations, generate_mean_and_std_dev_break_point, plot_observations_with_breaks
from utils.producer import setup_producer
from utils.logger import setup_logger

logger_settings = ProducerLoggerSettings()
logger = setup_logger(logger_settings)

producer_settings = KafkaProducerSettings()
producer = setup_producer(producer_settings)

logger.bind(event="producer_started",
            bootstrap_servers=producer_settings.BOOTSTRAP_SERVERS
            ).info("Kafka producer initialized")

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
        logger.bind(
                    event="distribution_shift",
                    count=count,
                    new_mean=mean,
                    new_std_dev=std_dev,
                    next_break_point=next_break_point
                    ).info("distribution parameters updated")
        list_mean.append(mean)
        list_breaks.append(next_break_point)

    obs = generate_observations(mean, std_dev)
    sent_timestamp = time.time()
    message = {"observation": obs, 
               "sent_timestamp": sent_timestamp, 
               "count": count}
    producer.send('test-topic', message)

    logger.bind(
                event="message_produced",
                observation=float(obs),
                sent_timestamp=sent_timestamp,
                count=count,
            ).info("message sent")
    
    time.sleep(5)
    count += 1
    list_observations.append(obs)
    
producer.flush()
logger.info("Break points: " + ", ".join(str(bp) for bp in list_breaks))

plot_observations_with_breaks(list_observations, list_breaks)


