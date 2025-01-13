FROM object-detection-base:latest

USER root

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH:-}" \
    MODEL_CACHE_DIR="/app/cache"

# Copy and install rtdetr requirements
COPY models/rtdetr/requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 3 --timeout 60 -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/cache /app/app/models/rtdetr && \
    chown -R appuser:appuser /app

# Switch to appuser after installations are complete
USER appuser

# Copy model code
COPY --chown=appuser:appuser models/rtdetr /app/app/models/rtdetr

# Make sure the models directory is a Python package
RUN touch /app/app/models/__init__.py
RUN touch /app/app/models/rtdetr/__init__.py

# Set the working directory
WORKDIR /app

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]