"""
config.py — All configuration from environment variables (12-factor app).
"""
import os

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "ecg-arrhythmia-topic")
KAFKA_GROUP_ID          = os.getenv("KAFKA_GROUP_ID", "ecg-arrhythmia-consumer-group")

# MongoDB
MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB         = os.getenv("MONGO_DB", "ecg_analysis")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "arrhythmia_results")

# Processing constants (fixed — matches Shreyas confirmation)
SAMPLING_RATE      = 125
SAMPLES_PER_MINUTE = 7500
WINDOW_SECONDS     = 10
WINDOW_SAMPLES     = SAMPLING_RATE * WINDOW_SECONDS  # 1250

# Consumer threading
CONSUMER_THREADS = int(os.getenv("CONSUMER_THREADS", "5"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Facility ID — set per deployment
FACILITY_ID = os.getenv("FACILITY_ID", "CF1315821527")
