
## Getting Started

Copy files within `/mlops` dir:

```bash
cp -i ../{pyproject.toml,uv.lock} .
```

Build image and run the container in detached mode:

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

CONTAINER ID   IMAGE                 COMMAND                  CREATED          STATUS          PORTS                    NAMES
2c6b0d0fcddd   mlops-app             "python3"                16 seconds ago   Up 15 seconds                            app
a5f3b442c34a   apache/kafka:latest   "/__cacert_entrypoin…"   16 seconds ago   Up 15 seconds   0.0.0.0:9092->9092/tcp   broker
```

Open a command terminal on the `mlops-app` container:

```bash
docker exec -it 2c6b0d0fcddd bash

root@2c6b0d0fcddd:/app# ls
consumer.py  producer.py
```



