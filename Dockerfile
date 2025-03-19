# Use NVIDIA PyTorch Image (CUDA + Python included)
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set working directory
WORKDIR /app

# Copy dependency file first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py logo.png number_plate_blur.pt ./

# Expose the port Flask runs on
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
