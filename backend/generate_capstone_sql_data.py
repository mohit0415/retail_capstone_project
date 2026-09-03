import random
from datetime import datetime, timedelta
import psycopg2

# ---------- CONFIG ----------
DB_CONFIG = {
    "dbname": "retail_compliance_db",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": 5432
}

NUM_VENDORS = 75
AUDITS_PER_VENDOR = 2
NUM_RETENTION = 60
NUM_REVIEWS = 60

random.seed(42)
# ----------------------------

def random_date(start_year=2023, end_year=2026):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def risk_category(score):
    if score >= 85:
        return "Critical"
    elif score >= 70:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

def severity_from_risk(score):
    if score >= 85:
        return random.choice(["High", "Critical"])
    elif score >= 70:
        return random.choice(["Medium", "High"])
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    vendor_ids = []

    # -------- Vendors --------
    for i in range(NUM_VENDORS):
        score = random.randint(40, 95)
        category = risk_category(score)

        cur.execute("""
            INSERT INTO vendors
            (vendor_name, risk_score, risk_category, compliance_status,
             approval_status, onboarding_date, last_audit_date, next_review_due)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING vendor_id
        """, (
            f"Vendor_{i}",
            score,
            category,
            random.choice(["Compliant", "Under Review", "Non-Compliant"]),
            random.choice(["Approved", "Pending", "Rejected"]),
            random_date(2022, 2024),
            random_date(2024, 2025),
            random_date(2025, 2026)
        ))

        vendor_ids.append(cur.fetchone()[0])

    # -------- Audit Logs --------
    for vid in vendor_ids:
        for _ in range(AUDITS_PER_VENDOR):
            score = random.randint(40, 95)
            severity = severity_from_risk(score)

            identified = random_date(2024, 2025)
            target = identified + timedelta(days=random.randint(15, 60))
            resolved = None

            remediation_status = random.choice(["Open", "In Progress", "Closed"])

            if remediation_status == "Closed":
                resolved = target - timedelta(days=random.randint(1, 10))

            escalation = remediation_status != "Closed" and datetime.now() > target

            cur.execute("""
                INSERT INTO audit_logs
                (vendor_id, policy_reference, issue_title, issue_severity,
                 remediation_status, issue_identified_date,
                 target_resolution_date, resolution_date, escalation_flag)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                vid,
                random.choice([
                    "Vendor Compliance Policy",
                    "Data Retention Policy",
                    "Access Control Policy",
                    "Anti-Bribery Policy"
                ]),
                "Generated Compliance Finding",
                severity,
                remediation_status,
                identified,
                target,
                resolved,
                escalation
            ))

    # -------- Retention Records --------
    for _ in range(NUM_RETENTION):
        cur.execute("""
            INSERT INTO retention_records
            (department, vendor_id, data_category,
             retention_period_years, legal_hold_flag,
             approval_status, last_review_date, next_review_due)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            random.choice(["Finance", "Marketing", "HR", "IT", "Legal"]),
            random.choice(vendor_ids),
            random.choice([
                "Transaction Records",
                "Customer Email Data",
                "Employee Records",
                "Security Logs",
                "Contract Documents"
            ]),
            random.randint(2, 10),
            random.choice([True, False]),
            random.choice(["Approved", "Pending"]),
            random_date(2024, 2025),
            random_date(2025, 2026)
        ))

    # -------- Compliance Reviews --------
    for _ in range(NUM_REVIEWS):
        cur.execute("""
            INSERT INTO compliance_reviews
            (vendor_id, reviewer_name, review_type,
             review_status, review_notes, review_date, next_review_due)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (
            random.choice(vendor_ids),
            random.choice(["Anita Sharma", "Rahul Mehta", "Priya Iyer", "Arjun Rao"]),
            random.choice([
                "Quarterly Review",
                "Annual Certification",
                "Escalation Review"
            ]),
            random.choice(["Open", "Closed", "In Progress"]),
            "Synthetic generated review",
            random_date(2024, 2025),
            random_date(2025, 2026)
        ))

    conn.commit()
    cur.close()
    conn.close()

    print("Capstone dataset generated successfully.")

if __name__ == "__main__":
    main()
