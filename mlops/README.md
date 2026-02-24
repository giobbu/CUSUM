
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

2026-02-19 16:09:53.125 | INFO     | __main__:<module>:15 - Sent: {'number': 0} at 2026-02-19 16:09:53.125780
2026-02-19 16:09:54.127 | INFO     | __main__:<module>:15 - Sent: {'number': 1} at 2026-02-19 16:09:54.127241
2026-02-19 16:09:55.129 | INFO     | __main__:<module>:15 - Sent: {'number': 2} at 2026-02-19 16:09:55.129416
2026-02-19 16:09:56.133 | INFO     | __main__:<module>:15 - Sent: {'number': 3} at 2026-02-19 16:09:56.133139
2026-02-19 16:09:57.136 | INFO     | __main__:<module>:15 - Sent: {'number': 4} at 2026-02-19 16:09:57.136886
2026-02-19 16:09:58.139 | INFO     | __main__:<module>:15 - Sent: {'number': 5} at 2026-02-19 16:09:58.139222
2026-02-19 16:09:59.141 | INFO     | __main__:<module>:15 - Sent: {'number': 6} at 2026-02-19 16:09:59.141634
2026-02-19 16:10:00.144 | INFO     | __main__:<module>:15 - Sent: {'number': 7} at 2026-02-19 16:10:00.144923
2026-02-19 16:10:01.146 | INFO     | __main__:<module>:15 - Sent: {'number': 8} at 2026-02-19 16:10:01.146403
2026-02-19 16:10:02.150 | INFO     | __main__:<module>:15 - Sent: {'number': 9} at 2026-02-19 16:10:02.150500
```

Open another terminal and execute `consumer.py`:

```bash
root@ab4b47be275a:/app# uv run consumer.py 
2026-02-19 16:10:11.019 | INFO     | __main__:<module>:15 - Received: {'number': 0} at 2026-02-19 16:10:11.019847
2026-02-19 16:10:11.019 | INFO     | __main__:<module>:15 - Received: {'number': 1} at 2026-02-19 16:10:11.019948
2026-02-19 16:10:11.019 | INFO     | __main__:<module>:15 - Received: {'number': 2} at 2026-02-19 16:10:11.019977
2026-02-19 16:10:11.020 | INFO     | __main__:<module>:15 - Received: {'number': 3} at 2026-02-19 16:10:11.020010
2026-02-19 16:10:11.020 | INFO     | __main__:<module>:15 - Received: {'number': 4} at 2026-02-19 16:10:11.020032
2026-02-19 16:10:11.020 | INFO     | __main__:<module>:15 - Received: {'number': 5} at 2026-02-19 16:10:11.020050
2026-02-19 16:10:11.020 | INFO     | __main__:<module>:15 - Received: {'number': 6} at 2026-02-19 16:10:11.020069
2026-02-19 16:10:11.020 | INFO     | __main__:<module>:15 - Received: {'number': 7} at 2026-02-19 16:10:11.020091
2026-02-19 16:10:11.020 | INFO     | __main__:<module>:15 - Received: {'number': 8} at 2026-02-19 16:10:11.020122
2026-02-19 16:10:11.020 | INFO     | __main__:<module>:15 - Received: {'number': 9} at 2026-02-19 16:10:11.020136
```

Open Prometheus UI at `http://0.0.0.0:9090/query` and Grafana at `http://0.0.0.0:3000/login`


**Debugging**
processes listening to port `:8000`
```bash
lsof -nP -i:8000 | grep LISTEN
```
