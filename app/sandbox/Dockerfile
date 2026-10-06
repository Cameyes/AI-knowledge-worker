# Build:  docker build -t rag-sandbox:py312-v1 app/sandbox
# The image is the ONLY place packages come from; there is no runtime pip and no network.
FROM python:3.12-slim

RUN pip install --no-cache-dir pandas==2.2.3 numpy==2.1.3 python-dateutil==2.9.0.post0

RUN mkdir -p /opt/runner /work
COPY runner.py /opt/runner/runner.py
RUN chmod 444 /opt/runner/runner.py && chown -R root:root /opt/runner

USER 65534:65534
WORKDIR /work
ENTRYPOINT ["python", "-I", "-B", "/opt/runner/runner.py"]
