FROM evalsysorg/mcpmark@sha256:05379ddb62790d9037a5521c46155b92304d95ae192caa76edfc5b8d2ac5474f
USER root
WORKDIR /opt/toolatlas
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN python3 -m venv .venv && .venv/bin/python -m pip install --no-cache-dir '.[dev,nim]' && chmod -R a+rX /opt/toolatlas
COPY benchmarks/docker ./benchmarks/docker
ADD --checksum=sha256:99d5449cef45bfcda6e5260f6ef4cd356bdbae59818d37ffa840054cdee19ec4 https://storage.mcpmark.ai/filesystem/file_property.zip /opt/snapshots/file_property.zip
RUN chmod 444 /opt/snapshots/file_property.zip
RUN mkdir -p /work && chown 10001:10001 /work
ENV PATH="/opt/toolatlas/.venv/bin:${PATH}" PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["/opt/toolatlas/.venv/bin/python", "/opt/toolatlas/benchmarks/docker/run.py"]
