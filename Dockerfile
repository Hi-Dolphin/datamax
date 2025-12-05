# Multi-stage build for DataMax
# Stage 1: Build stage
FROM python:3.11-slim-bookworm as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
    
# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . /app
WORKDIR /app

# Install the package
RUN pip install -e .

# Stage 2: Runtime stage
FROM python:3.11-slim-bookworm as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
    
# Create non-root user
RUN groupadd -r datamax && useradd -r -g datamax datamax

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/output && \
    chown -R datamax:datamax /app

# Switch to non-root user
USER datamax

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import datamax; print('DataMax is healthy')" || exit 1

# Expose port (if running web interface)
EXPOSE 8000

# Default command
CMD ["datamax", "--help"]

# Labels
LABEL maintainer="DataMax Team" \
      version="0.2.0" \
      description="DataMax - Advanced Data Crawling and Processing Framework" \
      org.opencontainers.image.title="DataMax" \
      org.opencontainers.image.description="Advanced data crawling and processing framework" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.vendor="DataMax Team" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/Hi-Dolphin/datamax" \
      org.opencontainers.image.documentation="https://github.com/Hi-Dolphin/datamax/blob/main/README.md"

# docker build --no-cache -t datamax:latest .
# docker run --rm --name datamax-worker -it datamax:latest /bin/bash python3 