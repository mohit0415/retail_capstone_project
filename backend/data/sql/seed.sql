TRUNCATE compliance_reviews, retention_records, vendors RESTART IDENTITY CASCADE;

INSERT INTO vendors (vendor_name, department, category, compliance_status, approval_status,
                     handles_personal_data, contract_start_date, contract_end_date, last_review_date)
VALUES
    ('Northgate Logistics', 'logistics', 'transport', 'compliant', 'approved', FALSE, '2023-04-01', '2026-03-31', '2025-09-12'),
    ('Northgate Payments', 'finance', 'payment_processing', 'under_review', 'pending', TRUE, '2024-01-15', '2026-01-14', '2025-07-30'),
    ('Meridian Cloud Services', 'it', 'hosting', 'compliant', 'approved', TRUE, '2022-06-01', '2025-05-31', '2025-04-18'),
    ('Halcyon Cleaning Co', 'facilities', 'facilities', 'compliant', 'approved', FALSE, '2024-02-01', '2027-01-31', '2025-11-02'),
    ('Vertex Marketing Partners', 'marketing', 'agency', 'non_compliant', 'approved', TRUE, '2023-09-01', '2026-08-31', '2025-10-05'),
    ('Ironwood Security Group', 'facilities', 'physical_security', 'compliant', 'approved', FALSE, '2023-01-01', '2025-12-31', '2025-06-21'),
    ('Sable Analytics', 'marketing', 'data_analytics', 'suspended', 'pending', TRUE, '2024-05-01', '2026-04-30', '2025-12-01'),
    ('Cobalt Staffing Solutions', 'hr', 'staffing', 'compliant', 'approved', TRUE, '2023-07-01', '2025-06-30', '2025-03-14'),
    ('Larkspur Print Works', 'marketing', 'print', 'compliant', 'approved', FALSE, '2024-03-01', '2026-02-28', '2025-08-09'),
    ('Quill Legal Advisory', 'legal', 'professional_services', 'compliant', 'approved', TRUE, '2022-11-01', '2026-10-31', '2025-05-27');

INSERT INTO retention_records (record_type, department, description, created_date, retention_until, disposal_status, legal_basis)
VALUES
    ('transaction_record', 'finance', 'Point of sale transactions Q1 2018', '2018-03-31', '2025-03-31', 'due', 'tax statute'),
    ('transaction_record', 'finance', 'Point of sale transactions Q2 2019', '2019-06-30', '2026-06-30', 'active', 'tax statute'),
    ('customer_profile', 'marketing', 'Loyalty programme members inactive since 2021', '2021-01-10', '2025-01-10', 'due', 'consent'),
    ('customer_profile', 'marketing', 'Loyalty programme members active', '2024-01-10', '2027-01-10', 'active', 'consent'),
    ('employee_record', 'hr', 'Terminated employee files 2019 cohort', '2019-12-31', '2025-12-31', 'active', 'employment law'),
    ('employee_record', 'hr', 'Terminated employee files 2017 cohort', '2017-12-31', '2023-12-31', 'due', 'employment law'),
    ('cctv_footage', 'facilities', 'Store 14 entrance recordings November 2025', '2025-11-01', '2025-12-01', 'due', 'legitimate interest'),
    ('cctv_footage', 'facilities', 'Store 09 entrance recordings December 2025', '2025-12-01', '2026-01-01', 'active', 'legitimate interest'),
    ('vendor_contract', 'legal', 'Terminated supplier agreements 2016', '2016-08-01', '2023-08-01', 'legal_hold', 'ongoing dispute'),
    ('access_log', 'it', 'Warehouse system access logs 2024', '2024-12-31', '2026-12-31', 'active', 'ISO 27001 A.8.15'),
    ('access_log', 'it', 'Warehouse system access logs 2022', '2022-12-31', '2024-12-31', 'due', 'ISO 27001 A.8.15'),
    ('incident_report', 'it', 'Security incident register 2023', '2023-12-31', '2028-12-31', 'active', 'ISO 27001 A.5.24'),
    ('customer_profile', 'marketing', 'Erasure requests received 2025', '2025-06-15', '2025-07-15', 'disposed', 'GDPR Art. 17'),
    ('transaction_record', 'finance', 'Refund ledger 2020', '2020-12-31', '2027-12-31', 'active', 'tax statute'),
    ('employee_record', 'hr', 'Payroll records 2021', '2021-12-31', '2028-12-31', 'active', 'employment law');

INSERT INTO compliance_reviews (vendor_id, review_date, reviewer, outcome, findings, remediation_due, remediation_completed)
VALUES
    (1, '2025-09-12', 'a.rao', 'passed', 'No findings. Transport only, no personal data processed.', NULL, NULL),
    (2, '2025-07-30', 'a.rao', 'conditional', 'Processor agreement missing Art. 28 sub-processor clause.', '2025-10-30', NULL),
    (3, '2025-04-18', 'j.mehta', 'passed', 'ISO 27001 certificate verified, valid to 2026-04.', NULL, NULL),
    (3, '2024-04-20', 'j.mehta', 'conditional', 'Encryption at rest not evidenced for backup tier.', '2024-07-20', '2024-07-11'),
    (5, '2025-10-05', 'j.mehta', 'failed', 'Customer email lists shared with an undisclosed sub-processor.', '2025-11-30', NULL),
    (6, '2025-06-21', 'a.rao', 'passed', 'Guard vetting records complete.', NULL, NULL),
    (7, '2025-12-01', 'p.singh', 'failed', 'Analytics pipeline retains raw identifiers beyond stated purpose.', '2026-01-15', NULL),
    (8, '2025-03-14', 'p.singh', 'passed', 'Right-to-work checks and data handling training evidenced.', NULL, NULL),
    (9, '2025-08-09', 'a.rao', 'passed', 'No personal data in scope.', NULL, NULL),
    (10, '2025-05-27', 'p.singh', 'conditional', 'Confidentiality annex predates current privacy policy version.', '2025-09-27', '2025-09-19');
