from confluent_kafka import Producer
from os import environ

TOPIC = environ.get("KAFKA_TOPIC", "conductor-response")
USER = environ.get("KAFKA_USER")
PASSWORD = environ.get("KAFKA_PASSWORD")
ADDRESS = environ.get("KAFKA_BROKER_ADDR")

error_list = []
if TOPIC is None:
    error_list.append("KAFKA_TOPIC")
if USER is None:
    error_list.append("KAFKA_USER")
if PASSWORD is None:
    error_list.append("KAFKA_PASSWORD")
if ADDRESS is None:
    error_list.append("KAFKA_BROKER_ADDR")
assert len(error_list) == 0, f"Environment variables {error_list} not set"

KAFKA_CONF = {
    "bootstrap.servers": ADDRESS,
    "client.id": "conductor-echo",
    "security.protocol": "SASL_PLAINTEXT",
    "sasl.mechanisms": "SCRAM-SHA-512",
    "sasl.username": USER,
    "sasl.password": PASSWORD,
}

producer = Producer(KAFKA_CONF)


def produce_message(s: str):
    producer.produce(TOPIC, value=s)
    producer.flush()