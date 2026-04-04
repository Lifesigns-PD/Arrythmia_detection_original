import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# ==========================================
# KAFKA CONFIGURATION
# ==========================================
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "ecg-arrhythmia-topic")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "ecg-arrhythmia-consumer-group")

# ==========================================
# MONGODB CONFIGURATION
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "ecg_analysis")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "arrhythmia_results")

# ==========================================
# SIGNAL PROCESSING CONSTANTS
# ==========================================
SAMPLING_RATE = 125
SAMPLES_PER_MINUTE = 7500
WINDOW_SECONDS = 10
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS 

# ==========================================
# SYSTEM
# ==========================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")