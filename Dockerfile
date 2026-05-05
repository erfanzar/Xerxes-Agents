# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY README.md .

# Create wheels
RUN pip install --upgrade pip wheel setuptools
RUN pip wheel --wheel-dir=/wheels .

# Final stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XERXES_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r xerxes && useradd -r -g xerxes xerxes

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links=/wheels xerxes && \
    rm -rf /wheels

# Copy application code
COPY --chown=xerxes:xerxes xerxes_agent/ ./xerxes_agent/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R xerxes:xerxes /app

# Switch to non-root user
USER xerxes

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import xerxes; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "xerxes"]

# Expose ports (if needed for API server)
EXPOSE 8000

# Labels
LABEL maintainer="erfanzar <Erfanzare810@gmail.com>" \
      version="0.0.18" \
      description="Xerxes - AI Agent Orchestration Framework"