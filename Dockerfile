# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies including OpenMP for XGBoost
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libomp-dev \
    libgomp1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create a non-root user for security and set up directories
RUN useradd -m -u 1000 jupyter && \
    mkdir -p /home/jupyter/.local/share/jupyter && \
    chown -R jupyter:jupyter /app /home/jupyter

USER jupyter

# Expose port for Jupyter notebook
EXPOSE 8888

# Set environment variables
ENV PYTHONPATH=/app
ENV JUPYTER_ENABLE_LAB=yes
ENV HOME=/home/jupyter

# Command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]