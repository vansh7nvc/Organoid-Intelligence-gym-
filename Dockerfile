# OrganoidEnv: Containerized Neuromorphic Simulation Testbed
FROM python:3.11-slim

# Install system compilation dependencies for Brian2 Cython backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository source
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Default command: run quickstart example
CMD ["python", "examples/01_quickstart.py"]
