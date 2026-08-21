# ---- Build Stage ----
FROM python:3.11-slim AS builder

# System dependencies for Brian2 Cython compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Runtime Stage ----
FROM python:3.11-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Default: run Month 6 Grand Unification training
CMD ["python", "experiments/12_month6_training.py"]
