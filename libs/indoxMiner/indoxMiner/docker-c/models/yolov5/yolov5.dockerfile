FROM indoxminer-base

# Install model-specific dependencies
COPY requirements/requirments-yolov5.txt .
RUN pip install --no-cache-dir -r requirements-detr.txt
# Copy model files
COPY models/detr.py /app/models/
COPY utils/ /app/utils/
COPY api.py /app/

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]