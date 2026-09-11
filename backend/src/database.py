import logging

import psycopg2

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "dbname": "retail_compliance_db",
    "user": "mohit",
    "password": "loadbalancer",
    "host": "localhost",
    "port": 5432
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def initialize_database():
    conn = get_db_connection()

    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM vendors")

        result = cur.fetchone()

        vendor_count = result[0]

        logger.info("existing vendors: %s", vendor_count)

        if vendor_count == 0:
            logger.info("database is empty, seeding data")

            from src.startup import generate_capstone_sql_data

            generate_capstone_sql_data()

        else:
            logger.info("data already exists, skipping seed")

        cur.close()

    finally:
        conn.close()
