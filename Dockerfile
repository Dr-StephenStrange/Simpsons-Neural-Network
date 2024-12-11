# Use the official Python 3.11 slim image as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to force TensorFlow to use CPU and suppress logs
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_CPP_MIN_LOG_LEVEL="2"

# Copy the requirements file to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files to the container
COPY . .

# Expose the default Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "gradio_app.py"]
