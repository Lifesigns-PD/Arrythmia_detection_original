import json
import logging
from confluent_kafka import Consumer, KafkaError
from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_GROUP_ID
from ecg_processor import process

# Setup basic logging for stdout (Kubernetes best practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def start_consumer():
    # 1. Configure the Kafka Consumer
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'earliest' # Read from the beginning if we missed anything
    }

    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])
    logging.info(f"Connected to Kafka! Listening to topic: {KAFKA_TOPIC}...")

    # 2. Start the infinite listening loop
    try:
        while True:
            msg = consumer.poll(timeout=1.0) # Check for a message every 1 second

            if msg is None:
                continue # No message, keep waiting
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logging.error(f"Kafka Error: {msg.error()}")
                    break

            # 3. We got a message! Parse the JSON
            try:
                payload = json.loads(msg.value().decode('utf-8'))
                device_id = payload.get("deviceId", "UNKNOWN")
                admission_id = payload.get("admissionId", "UNKNOWN")
                timestamp = payload.get("timestamp", 0)
                ecg_data = payload.get("ecgData", [])

                logging.info(f"Received ECG batch for Admission: {admission_id} | {len(ecg_data)} samples")

                # 4. Pass it to your processor (the AI/Rules brain!)
                result = process(ecg_data, admission_id, device_id, timestamp)
                
                logging.info(f"Successfully processed! Generated UUID: {result['uuid']}")
                
                # NOTE: Next step will be writing this result to MongoDB!
                
            except Exception as e:
                logging.error(f"Failed to process message: {e}")

    except KeyboardInterrupt:
        logging.info("Shutting down consumer safely...")
    finally:
        # 5. Close down gracefully on exit (Crucial for Kubernetes)
        consumer.close()

if __name__ == "__main__":
    start_consumer()