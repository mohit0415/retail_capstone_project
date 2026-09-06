TRUNCATE compliance_reviews, retention_records, audit_logs, vendors RESTART IDENTITY CASCADE;

INSERT INTO vendors (vendor_name, risk_score, risk_category, compliance_status,
                     approval_status, onboarding_date, last_audit_date, next_review_due)
VALUES
    ('Northgate Logistics',        44, 'Low',      'Compliant',     'Approved', '2023-04-01', '2025-06-12', '2026-06-12'),
    ('Northgate Payments',         78, 'High',     'Under Review',  'Pending',  '2024-01-15', '2025-07-30', '2025-11-30'),
    ('Meridian Cloud Services',    72, 'High',     'Compliant',     'Approved', '2022-06-01', '2025-04-18', '2026-04-18'),
    ('Halcyon Cleaning Co',        41, 'Low',      'Compliant',     'Approved', '2024-02-01', '2025-11-02', '2026-11-02'),
    ('Vertex Marketing Partners',  88, 'Critical', 'Non-Compliant', 'Approved', '2023-09-01', '2025-10-05', '2025-12-05'),
    ('Ironwood Security Group',    57, 'Medium',   'Compliant',     'Approved', '2023-01-01', '2025-06-21', '2026-06-21'),
    ('Sable Analytics',            91, 'Critical', 'Non-Compliant', 'Rejected', '2024-05-01', '2025-09-01', '2025-11-01'),
    ('Cobalt Staffing Solutions',  63, 'Medium',   'Under Review',  'Approved', '2023-07-01', '2025-03-14', '2026-03-14'),
    ('Larkspur Print Works',       46, 'Low',      'Compliant',     'Approved', '2024-03-01', '2025-08-09', '2026-08-09'),
    ('Quill Legal Advisory',       52, 'Medium',   'Compliant',     'Approved', '2022-11-01', '2025-05-27', '2025-11-27'),
    ('Perrin Cold Chain',          69, 'Medium',   'Under Review',  'Pending',  '2024-08-12', '2025-08-20', '2026-02-20'),
    ('Aldwych Facilities Group',   83, 'High',     'Non-Compliant', 'Approved', '2023-02-20', '2025-02-28', '2025-08-28');

INSERT INTO audit_logs (vendor_id, policy_reference, issue_title, issue_severity,
                        remediation_status, issue_identified_date, target_resolution_date,
                        resolution_date, escalation_flag)
VALUES
    (2,  'Supplier Vendor Compliance Policy', 'Processor agreement missing sub-processor clause', 'High',     'In Progress', '2025-07-30', '2025-10-30', NULL,         TRUE),
    (2,  'Retail Data Protection Privacy Policy', 'Transfer impact assessment not evidenced',     'Medium',   'Open',        '2025-09-14', '2025-12-14', NULL,         TRUE),
    (3,  'ISO 27001 Access Control Summary', 'Encryption at rest not evidenced for backup tier',  'Medium',   'Closed',      '2024-04-20', '2024-07-20', '2024-07-11', FALSE),
    (5,  'Anti Bribery Ethical Conduct Policy', 'Hospitality register not maintained',            'Critical', 'Open',        '2025-10-05', '2025-11-20', NULL,         TRUE),
    (5,  'Supplier Vendor Compliance Policy', 'Annual certification lapsed',                      'High',     'In Progress', '2025-08-11', '2025-11-11', NULL,         TRUE),
    (6,  'Information Security Access Control Policy', 'Shared account used for site access',     'Medium',   'Closed',      '2025-03-02', '2025-05-02', '2025-04-27', FALSE),
    (7,  'Retail Data Protection Privacy Policy', 'Personal data exported without a lawful basis','Critical', 'Open',        '2025-09-01', '2025-10-15', NULL,         TRUE),
    (7,  'GDPR Selected Articles', 'Erasure request unanswered past one month',                   'Critical', 'In Progress', '2025-10-02', '2025-11-02', NULL,         TRUE),
    (8,  'Data Retention and Archival Policy', 'Candidate records held past retention period',    'Medium',   'Open',        '2025-06-18', '2026-01-18', NULL,         FALSE),
    (10, 'Anti Bribery Ethical Conduct Policy', 'Conflict of interest declaration missing',       'Low',      'Closed',      '2025-05-27', '2025-08-27', '2025-08-01', FALSE),
    (11, 'Supplier Vendor Compliance Policy', 'Temperature excursion log incomplete',             'Medium',   'Open',        '2025-08-20', '2026-02-20', NULL,         FALSE),
    (12, 'Information Security Access Control Policy', 'Leaver access not revoked within 24 hours','High',    'Open',        '2025-02-28', '2025-05-28', NULL,         TRUE),
    (12, 'ISO 27001 Access Control Summary', 'Privileged access review overdue two cycles',       'High',     'In Progress', '2025-06-30', '2025-09-30', NULL,         TRUE);

INSERT INTO retention_records (vendor_id, department, data_category, retention_period_years,
                               legal_hold_flag, approval_status, last_review_date, next_review_due)
VALUES
    (1,  'Finance',   'Transaction Records',   7,  FALSE, 'Approved', '2025-01-15', '2026-01-15'),
    (1,  'Legal',     'Contract Documents',    10, FALSE, 'Approved', '2025-02-10', '2026-02-10'),
    (2,  'Finance',   'Transaction Records',   7,  FALSE, 'Approved', '2024-11-30', '2025-11-30'),
    (2,  'Legal',     'Contract Documents',    10, TRUE,  'Approved', '2025-03-05', '2026-03-05'),
    (3,  'IT',        'Security Logs',         2,  FALSE, 'Approved', '2025-04-18', '2026-04-18'),
    (3,  'IT',        'Customer Email Data',   3,  FALSE, 'Pending',  '2024-10-01', '2025-10-01'),
    (4,  'Finance',   'Transaction Records',   7,  FALSE, 'Approved', '2025-05-20', '2026-05-20'),
    (5,  'Marketing', 'Customer Email Data',   3,  FALSE, 'Pending',  '2024-09-12', '2025-09-12'),
    (5,  'Marketing', 'Transaction Records',   7,  FALSE, 'Approved', '2025-06-01', '2026-06-01'),
    (6,  'IT',        'Security Logs',         2,  FALSE, 'Approved', '2025-06-21', '2026-06-21'),
    (7,  'Marketing', 'Customer Email Data',   3,  FALSE, 'Pending',  '2024-08-15', '2025-08-15'),
    (7,  'Legal',     'Contract Documents',    10, TRUE,  'Approved', '2025-07-01', '2026-07-01'),
    (8,  'HR',        'Employee Records',      6,  FALSE, 'Approved', '2024-12-14', '2025-12-14'),
    (8,  'HR',        'Employee Records',      6,  TRUE,  'Approved', '2025-01-30', '2026-01-30'),
    (9,  'Marketing', 'Customer Email Data',   3,  FALSE, 'Approved', '2025-08-09', '2026-08-09'),
    (10, 'Legal',     'Contract Documents',    10, FALSE, 'Approved', '2025-05-27', '2026-05-27'),
    (11, 'Finance',   'Transaction Records',   7,  FALSE, 'Pending',  '2024-10-20', '2025-10-20'),
    (12, 'IT',        'Security Logs',         2,  FALSE, 'Approved', '2024-09-28', '2025-09-28');

INSERT INTO compliance_reviews (vendor_id, reviewer_name, review_type, review_status,
                                review_notes, review_date, next_review_due)
VALUES
    (1,  'Anita Sharma', 'Annual Certification', 'Closed',      'No findings. Transport only, no personal data processed.',   '2025-06-12', '2026-06-12'),
    (2,  'Anita Sharma', 'Quarterly Review',     'In Progress', 'Processor agreement gap tracked under an open finding.',     '2025-07-30', '2025-11-30'),
    (2,  'Rahul Mehta',  'Escalation Review',    'Open',        'Escalated after the remediation deadline passed.',           '2025-11-01', '2025-12-15'),
    (3,  'Rahul Mehta',  'Annual Certification', 'Closed',      'ISO 27001 certificate verified, valid to 2026-04.',          '2025-04-18', '2026-04-18'),
    (4,  'Priya Iyer',   'Annual Certification', 'Closed',      'Facilities scope only. No system access granted.',           '2025-11-02', '2026-11-02'),
    (5,  'Priya Iyer',   'Escalation Review',    'Open',        'Two critical findings open past target. Board notified.',    '2025-10-05', '2025-12-05'),
    (6,  'Arjun Rao',    'Quarterly Review',     'Closed',      'Shared account remediated and verified.',                    '2025-06-21', '2026-06-21'),
    (7,  'Arjun Rao',    'Escalation Review',    'Open',        'Suspended pending lawful basis review for exported data.',   '2025-09-01', '2025-11-01'),
    (8,  'Anita Sharma', 'Quarterly Review',     'In Progress', 'Retention overrun under review with HR.',                    '2025-03-14', '2026-03-14'),
    (9,  'Rahul Mehta',  'Annual Certification', 'Closed',      'Print only. No personal data retained after fulfilment.',    '2025-08-09', '2026-08-09'),
    (10, 'Priya Iyer',   'Annual Certification', 'Closed',      'Declaration obtained and filed.',                            '2025-05-27', '2025-11-27'),
    (11, 'Arjun Rao',    'Quarterly Review',     'In Progress', 'Cold chain logging gap raised with the supplier.',           '2025-08-20', '2026-02-20'),
    (12, 'Rahul Mehta',  'Escalation Review',    'Open',        'Leaver access finding unresolved across two review cycles.', '2025-02-28', '2025-08-28');
