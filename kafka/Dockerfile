# Use a lightweight, official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (this makes rebuilding faster)
COPY requirements.txt .

# Install the Python dependencies (and basic C++ compilers needed for scipy/Kafka)
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean

# Copy the rest of your application code into the container
COPY . .

# Command to run the application (The infinite listener)
CMD ["python", "kafka_consumer.py"]