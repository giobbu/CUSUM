
## Getting Started

From within `/mlops` dir copy `source/detector/*` and `uv` files:

```bash
mkdir app/detector
```

```bash
cp -i -r ../source/detector/* ./app/detector
```

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

2026-03-02 17:53:44.943 | INFO     | __main__:<module>:34 - Sent: {'observation': -7.975584007894333, 'sent_timestamp': 1772474024.9425344, 'count': 1}

2026-03-02 17:53:49.945 | INFO     | __main__:<module>:34 - Sent: {'observation': -8.119080371260614, 'sent_timestamp': 1772474029.9447057, 'count': 2}
```

Open another terminal and execute `consumer.py`:

```bash
root@ab4b47be275a:/app# uv run consumer.py 

# message received
2026-03-02 17:53:44.948 | INFO     | __main__:<module>:42 - 
Value: -7.975584007894333, 
Sent at: 1772474024.9425344, 
Received at: 1772474024.9481857, 
Delay: 0.01 seconds
# PH-Test
2026-03-02 17:53:44.948 | INFO     | __main__:<module>:49 - 
PH-Test Detector Results: 
Positive Change: 0, 
Negative Change: 0, 
Change Detected: False, 
Metadata: 
 --- PH_CUSUM_Detector(warmup_period=30, delta=0.5, threshold=2)

# message received
2026-03-02 17:53:49.950 | INFO     | __main__:<module>:42 - 
Value: -8.119080371260614, 
Sent at: 1772474029.9447057, 
Received at: 1772474029.9503043, 
Delay: 0.01 seconds
# PH-Tests
2026-03-02 17:53:49.950 | INFO     | __main__:<module>:49 - 
PH-Test Detector Results: 
Positive Change: 0, 
Negative Change: 0, 
Change Detected: False, 
Metadata: 
 --- PH_CUSUM_Detector(warmup_period=30, delta=0.5, threshold=2)
```

## Open in Browser

- **Prometheus UI**: http://localhost:9090/query  
- **Grafana**: http://localhost:3000/login

## Debugging

### Check processes listening on port `8000`

```bash
lsof -nP -i:8000 | grep LISTEN
```

### Kill process using port `8000` (if needed)

```bash
kill -9 <PID>
```

Replace `<PID>` with the process ID returned from the `lsof` command.
