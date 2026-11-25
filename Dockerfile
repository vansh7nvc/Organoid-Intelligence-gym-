# OrganoidEnv: Reproducible Containerized SNN-RL Testbed & 3D Workbench
FROM python:3.11-slim

# System build tools for C99 SNN integration kernel & Brian2 Cython
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python runtime + platform + publication dependencies
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository source
COPY . .

# Install OrganoidEnv with platform and publication extras, and pre-compile C99 SNN kernel
RUN pip install --no-cache-dir -e ".[dev,publication,platform]" && \
    python -c "from organoid_rl.environment.fast_snn import load_snn_kernel; assert load_snn_kernel() is not None"

EXPOSE 8000

# Default: start interactive 3D workbench on 0.0.0.0:8000
CMD ["python", "scripts/launch_platform.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
