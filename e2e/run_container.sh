DOCKER_IMAGE="${DOCKER_IMAGE:-vllm-cpu-x86:latest}"
# Mount point for LLM model weights — set MODEL_DIR on the host before running.
# e.g.  MODEL_DIR=/home/ubuntu/models ./run_container.sh
MODEL_DIR="${MODEL_DIR:-/home/ubuntu/}"

sudo docker run --privileged -it --rm \
    -u root \
    --ipc=host --net=host --cap-add=ALL \
    -v "${PWD}":/workspace \
    -v "${MODEL_DIR}":/data \
    --entrypoint /bin/bash \
    ${DOCKER_IMAGE}
