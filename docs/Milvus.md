
### Step 1: Install Docker
1. **Install Docker** if you don’t have it yet.
   - Follow the official Docker installation guide for your OS:
     - [Windows](https://docs.docker.com/desktop/install/windows-install/)
     - [MacOS](https://docs.docker.com/desktop/install/mac-install/)
     - [Linux](https://docs.docker.com/engine/install/)

   


### Verify 
- **Command to Verify Installation:**
  ```bash
  docker --version
  ```
- **Expected Output:**
  ```bash
  Docker version 20.10.7, build f0df350
  ```
- **Error Handling:**
  - **Docker Command Not Found:**
    - Ensure Docker is installed. Follow the installation guide for your OS.
    - Reinstall Docker if necessary.
  - **Docker Daemon Not Running:**
    - Start Docker service:
      ```bash
      sudo systemctl start docker
      ```
    - Verify Docker is running:
      ```bash
      sudo systemctl status docker
      ```

#### 2. Install Docker Compose (if needed)
- **Command to Verify Installation:**
  ```bash
  docker-compose --version
  ```
- **Expected Output:**
  ```bash
  docker-compose version 1.29.2, build 5becea4c
  ```
- **Error Handling:**
  - **Docker Compose Command Not Found:**
    - Install Docker Compose by following the [installation guide](https://docs.docker.com/compose/install/).
  - **Permission Issues:**
    - Run commands with `sudo` if necessary:
      ```bash
      sudo docker-compose --version
      ```

#### 3. Clone the Milvus Repository
- **Command:**
  ```bash
  git clone https://github.com/milvus-io/milvus.git
  ```
- **Expected Output:**
  ```bash
  Cloning into 'milvus'...
  remote: Enumerating objects: 1234, done.
  remote: Counting objects: 100% (1234/1234), done.
  remote: Compressing objects: 100% (567/567), done.
  remote: Total 1234 (delta 678), reused 1234 (delta 678), pack-reused 0
  Receiving objects: 100% (1234/1234), 123.45 MiB | 12.34 MiB/s, done.
  Resolving deltas: 100% (678/678), done.
  ```
- **Error Handling:**
  - **Git Command Not Found:**
    - Ensure Git is installed:
      ```bash
      sudo apt-get install git
      ```
  - **Network Issues:**
    - Check your internet connection.
    - Verify the repository URL.

#### 4. Change Directory to Docker Deployment
- **Command:**
  ```bash
  cd milvus/deployments/docker
  ```
- **Error Handling:**
  - **Directory Not Found:**
    - Verify that the repository was cloned successfully.
    - Check the path for typos.

#### 5. Pull Milvus Docker Images
- **Command:**
  ```bash
  docker-compose pull
  ```
- **Expected Output:**
  ```bash
  Pulling milvus     ... done
  Pulling etcd       ... done
  Pulling minio      ... done
  Pulling pulsar     ... done
  ```
- **Error Handling:**
  - **Image Pull Failure:**
    - Check network connectivity.
    - Verify Docker Hub is reachable.
    - Retry pulling images:
      ```bash
      docker-compose pull
      ```
  - **Authentication Issues:**
    - Ensure you are logged into Docker Hub:
      ```bash
      docker login
      ```

#### 6. Start Milvus with Docker Compose
- **Command:**
  ```bash
  docker-compose up -d
  ```
- **Expected Output:**
  ```bash
  Creating network "docker_default" with the default driver
  Creating volume "docker_minio" with default driver
  Creating volume "docker_pulsar" with default driver
  Creating milvus ... done
  Creating etcd   ... done
  Creating minio  ... done
  Creating pulsar ... done
  ```
- **Error Handling:**
  - **Containers Not Starting:**
    - Check the status of the containers:
      ```bash
      docker-compose ps
      ```
    - View logs for detailed error messages:
      ```bash
      docker-compose logs
      ```
  - **Port Conflicts:**
    - Check if ports are already in use:
      ```bash
      sudo lsof -i :19530
      sudo kill -9 <PID>
      ```
    - Modify ports in `docker-compose.yml` if necessary.

#### 7. Check Running Containers
- **Command to Check Container Status:**
  ```bash
  docker-compose ps
  ```
- **Expected Output:**
  ```bash
  Name                 Command                  State                 Ports
  ---------------------------------------------------------------------------------
  docker_etcd_1         etcd --name etcd --data  Up      2379/tcp, 2380/tcp
  docker_minio_1        /usr/bin/docker-entrpoi  Up      9000/tcp
  docker_milvus_1       /bin/sh -c 'milvus-ser  Up      0.0.0.0:19530->19530/tcp
  docker_pulsar_1       /pulsar/bin/pulsar ser  Up      6650/tcp, 8080/tcp
  ```
- **Error Handling:**
  - **Containers Not Running:**
    - Check logs for errors:
      ```bash
      docker-compose logs
      ```
    - Ensure no errors during startup.

#### 8. Test Milvus Connection with Python SDK
- **Python Code to Test Connection:**
  ```python
  from pymilvus import connections

  try:
      connections.connect(host='127.0.0.1', port='19530')
      print("Connection successful")
  except Exception as e:
      print(f"Connection failed: {e}")
  ```
- **Expected Output:**
  ```bash
  Connection successful
  ```
- **Error Handling:**
  - **Connection Failed:**
    - Verify Milvus is running and accessible at the specified host and port.
    - Check container logs:
      ```bash
      docker-compose logs milvus
      ```

#### 9. Stop Milvus
- **Command:**
  ```bash
  docker-compose down
  ```
- **Expected Output:**
  ```bash
  Stopping pulsar  ... done
  Stopping minio   ... done
  Stopping etcd    ... done
  Stopping milvus  ... done
  Removing pulsar  ... done
  Removing minio   ... done
  Removing etcd    ... done
  Removing milvus  ... done
  Removing network docker_default
  ```
- **Error Handling:**
  - **Containers Not Stopping:**
    - Manually stop containers:
      ```bash
      docker stop <container_id>
      ```

#### 10. Remove Containers and Volumes
- **Command:**
  ```bash
  docker-compose down -v
  ```
- **Expected Output:**
  ```bash
  Stopping pulsar  ... done
  Stopping minio   ... done
  Stopping etcd    ... done
  Stopping milvus  ... done
  Removing pulsar  ... done
  Removing minio   ... done
  Removing etcd    ... done
  Removing milvus  ... done
  Removing network docker_default
  Removing volume docker_minio
  Removing volume docker_pulsar
  ```
- **Error Handling:**
  - **Volumes Not Removed:**
    - List and remove volumes manually if needed:
      ```bash
      docker volume ls
      docker volume rm <volume_name>
      ```

