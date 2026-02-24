
## Getting Started

From within `/mlops` dir copy `uv` files:

```bash
cp -i ../{pyproject.toml,uv.lock} ./app
```

Add `kafka-python` module from within `/app`:

```bash
uv add kafka-python
```

Add `prometheus-client` module
```bash
uv add prometheus-client
```

From within `/mlops` build image and run the container in detached mode:

```bash
docker-compose up -d

 ✔ app                    Built                                                                                                                                                                                                           0.0s 
 ✔ Network mlops_default  Created                                                                                                                                                                                                         0.0s 
 ✔ Container broker       Started                                                                                                                                                                                                         0.3s 
 ✔ Container app          Started
 ```

List running containers:

```bash
docker container list

CONTAINER ID   IMAGE                 COMMAND                  CREATED        STATUS        PORTS                    NAMES
4c0745216a04   grafana/grafana       "/run.sh"                27 hours ago   Up 27 hours   0.0.0.0:3000->3000/tcp   grafana
311bb81142c2   prom/prometheus       "/bin/prometheus --c…"   27 hours ago   Up 27 hours   0.0.0.0:9090->9090/tcp   prometheus
ab1ac2bdef46   mlops-app             "python3"                27 hours ago   Up 27 hours   0.0.0.0:8000->8000/tcp   app
6f1717e4ba88   apache/kafka:latest   "/__cacert_entrypoin…"   27 hours ago   Up 27 hours   0.0.0.0:9092->9092/tcp   broker
```

Open a command terminal on the `mlops-app` container:

```bash
docker exec -it 2c6b0d0fcddd bash

root@2c6b0d0fcddd:/app# ls
consumer.py  producer.py
```

Execute `producer.py`:

```bash
root@ab4b47be275a:/app# uv run producer.py 

2026-02-24 17:12:46.445 | INFO     | __main__:<module>:18 - Sent: {'number': 76, 'sent_timestamp': 1771953166.351782}
2026-02-24 17:12:47.450 | INFO     | __main__:<module>:18 - Sent: {'number': 44, 'sent_timestamp': 1771953167.4496365}
2026-02-24 17:12:48.452 | INFO     | __main__:<module>:18 - Sent: {'number': 10, 'sent_timestamp': 1771953168.4521246}
2026-02-24 17:12:49.458 | INFO     | __main__:<module>:18 - Sent: {'number': 62, 'sent_timestamp': 1771953169.457914}
2026-02-24 17:12:50.462 | INFO     | __main__:<module>:18 - Sent: {'number': 41, 'sent_timestamp': 1771953170.461526}
```

Open another terminal and execute `consumer.py`:

```bash
root@ab4b47be275a:/app# uv run consumer.py 

2026-02-24 17:12:46.447 | INFO     | __main__:<module>:28 - 
                     Value: 76, 
                     Sent at: 1771953166.351782, 
                     Received at: 1771953166.4472306, 
                     Delay: 0.10 seconds
2026-02-24 17:12:47.455 | INFO     | __main__:<module>:28 - 
                     Value: 44, 
                     Sent at: 1771953167.4496365, 
                     Received at: 1771953167.4554284, 
                     Delay: 0.01 seconds
2026-02-24 17:12:48.460 | INFO     | __main__:<module>:28 - 
                     Value: 10, 
                     Sent at: 1771953168.4521246, 
                     Received at: 1771953168.4605381, 
                     Delay: 0.01 seconds
2026-02-24 17:12:49.463 | INFO     | __main__:<module>:28 - 
                     Value: 62, 
                     Sent at: 1771953169.457914, 
                     Received at: 1771953169.4639313, 
                     Delay: 0.01 seconds
2026-02-24 17:12:50.468 | INFO     | __main__:<module>:28 - 
                     Value: 41, 
                     Sent at: 1771953170.461526, 
                     Received at: 1771953170.4681072, 
                     Delay: 0.01 seconds
```

Open Prometheus UI at `http://0.0.0.0:9090/query` and Grafana at `http://0.0.0.0:3000/login`


**Debugging**
processes listening to port `:8000`
```bash
lsof -nP -i:8000 | grep LISTEN
```
