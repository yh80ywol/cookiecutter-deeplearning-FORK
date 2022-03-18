from prefect import task, Flow
from prefect.tasks.docker import (CreateContainer, StartContainer) 

#für Windows!
docker_server_url = "npipe:////./pipe/docker_engine"

#Input arguments (TTY = allocate a pseudo-TTY connected to the container’s stdin, port = ...)
kwargs = dict(tty=True,ports=[80,80])

create = CreateContainer(image_name="prefecthq/prefect", extra_docker_kwargs = kwargs , docker_server_url = docker_server_url)
start = StartContainer(docker_server_url = docker_server_url)

with Flow("docker-flow") as flow:
    container_id = create()
    s = start(container_id = container_id)

#flow.run()
flow.register(project_name="test")