
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
<container-id>   mlops-app             "python3"                27 hours ago   Up 27 hours   0.0.0.0:8000->8000/tcp   app
6f1717e4ba88   apache/kafka:latest   "/__cacert_entrypoin…"   27 hours ago   Up 27 hours   0.0.0.0:9092->9092/tcp   broker
```

Open a command terminal on the `mlops-app` container:

```bash
docker exec -it <container-id> bash

root@<container-id>:/app# ls
consumer.py  producer.py
```

Execute `producer.py`:

```bash
root@<container-id>:/app# uv run producer.py 
```

Open another terminal and execute `consumer.py`:

```bash
root@<container-id>:/app# uv run consumer.py 
```

```bash
root@c<container-id>:/app# cd logs

root@c<container-id>:/app/logs# tail -f consumer.jsonl | grep '"change_detection_result"'

{"timestamp": 1772718547.277965, "level": "INFO", "service": "kafka-consumer", "host": "c730b105b417", "event": "change_detection_result", "message": "CUSUM change detection result", "observation": -5.537012962870473, "positive_increase": 0, "negative_increase": 0, "is_change": false, "detector_params": {"warmup_period": 30, "delta": 0.5, "threshold": 2}}
{"timestamp": 1772718552.281865, "level": "INFO", "service": "kafka-consumer", "host": "c730b105b417", "event": "change_detection_result", "message": "CUSUM change detection result", "observation": -5.923583643696739, "positive_increase": 0, "negative_increase": 0, "is_change": false, "detector_params": {"warmup_period": 30, "delta": 0.5, "threshold": 2}}
{"timestamp": 1772718557.285472, "level": "INFO", "service": "kafka-consumer", "host": "c730b105b417", "event": "change_detection_result", "message": "CUSUM change detection result", "observation": -4.7693634789138315, "positive_increase": 0, "negative_increase": 0, "is_change": false, "detector_params": {"warmup_period": 30, "delta": 0.5, "threshold": 2}}
{"timestamp": 1772718562.292707, "level": "INFO", "service": "kafka-consumer", "host": "c730b105b417", "event": "change_detection_result", "message": "CUSUM change detection result", "observation": -4.969309583579554, "positive_increase": 0, "negative_increase": 0, "is_change": false, "detector_params": {"warmup_period": 30, "delta": 0.5, "threshold": 2}}
{"timestamp": 1772718567.296737, "level": "INFO", "service": "kafka-consumer", "host": "c730b105b417", "event": "change_detection_result", "message": "CUSUM change detection result", "observation": -4.555649305852855, "positive_increase": 0, "negative_increase": 0, "is_change": false, "detector_params": {"warmup_period": 30, "delta": 0.5, "threshold": 2}}
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
