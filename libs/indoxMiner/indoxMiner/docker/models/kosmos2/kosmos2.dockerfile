FROM indoxminer-base

# Install model-specific dependencies
COPY requirements/requirements-kosmos2.txt . 
RUN pip install --no-cache-dir -r requirements/requirements-kosmos2.txt

# Create model cache directory
RUN mkdir -p /app/cache && chown appuser:appuser /app/cache

# Copy application files
COPY --chown=appuser:appuser models/detr.py /app/models/
COPY --chown=appuser:appuser utils/ /app/utils/
COPY --chown=appuser:appuser api.py /app/

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set cache directory environment variable
ENV MODEL_CACHE_DIR=/app/cache

# Define the command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
