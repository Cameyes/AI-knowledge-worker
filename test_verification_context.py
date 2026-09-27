from app.rag.verification_context import compact_evidence


test_cases = [
    {
        "query": "What was David Kim's 2023 performance rating?",
        "document": """
# David Kim

Employee Information
Employee ID: EMP-1042
Department: Platform Engineering
Location: Bangalore
Joined: 2021

2022 Performance
Performance rating: 4.1/5
Manager feedback: Strong technical contribution.

2022 Compensation
Base salary: $105,000
Bonus: $9,000

2023 Performance
Performance rating: 4.5/5
Manager feedback: Consistently delivered critical infrastructure projects.

2023 Compensation
Base salary: $110,000
Bonus: $12,000

2023 Projects
Led multiple internal platform improvements and reliability initiatives.

2024 Goals
Improve observability and reduce deployment times.
"""
    },

    {
        "query": "What infrastructure initiative did David Kim lead?",
        "document": """
# David Kim

Career History
Joined the company in 2021 as a Software Engineer.
Promoted to Senior Software Engineer in 2023.

Infrastructure
David Kim led the migration of the platform to Kubernetes,
improving system scalability and reliability.
The migration was completed in Q3 2023.

Application Development
Contributed to several internal services written in Python and Java.

Certifications
AWS Solutions Architect Associate.
Certified Kubernetes Administrator.

Performance
Received a 4.5/5 performance rating in 2023.

Recognition
Received an engineering excellence award for platform reliability.
"""
    },

    {
        "query": "What certifications did the employee obtain in 2023?",
        "document": """
# Employee Profile

Personal Information
Employee: Michael Carter
Department: Engineering
Location: Hyderabad

2022 Training
Completed internal security and cloud fundamentals training.

2022 Performance
Performance rating: 4.0/5

2023 Certifications
Obtained AWS Solutions Architect Associate certification.
Obtained Certified Kubernetes Administrator certification.

2023 Projects
Worked on internal applications and deployment automation.

2023 Performance
Performance rating: 4.3/5

2024 Development Plan
Focus on distributed systems and observability.
"""
    },

    {
        "query": "What is the supplier delivery requirement?",
        "document": """
# Supply Agreement

Supplier
Acme Industrial Systems

Payment Terms
Payment must be made within 30 days of invoice receipt.

Delivery
The supplier shall deliver all equipment within 30 days
of receiving the purchase order.
Delivery must be completed at the designated warehouse.

Inspection
All equipment is subject to inspection upon delivery.

Warranty
The equipment includes a two-year warranty covering manufacturing defects.

Penalties
Late delivery may result in contractual penalties.

Renewal
The agreement is automatically renewed annually unless terminated.
"""
    },

    # Harder multi-condition case
    {
        "query": "What was David Kim's 2023 bonus and what infrastructure project did he lead?",
        "document": """
# David Kim

Employee Information
Employee ID: EMP-1042
Department: Platform Engineering

2022 Compensation
Base salary: $105,000
Bonus: $9,000

2023 Performance
Performance rating: 4.5/5

Infrastructure
David Kim led the migration of the company's primary platform
to Kubernetes to improve scalability and reliability.

2023 Compensation
Base salary: $110,000
Bonus: $12,000

2023 Projects
Worked on deployment automation and observability improvements.

Certifications
AWS Solutions Architect Associate.
"""
    },

    # Multiple similar entities / strong distractor
    {
        "query": "What was the performance rating of David Kim in 2023?",
        "document": """
# Employee Records

## David Kim

2022 Performance
Performance rating: 4.1/5

2023 Performance
Performance rating: 4.5/5

2023 Bonus
Bonus: $12,000

## James Wilson

2022 Performance
Performance rating: 4.3/5

2023 Performance
Performance rating: 4.9/5

2023 Bonus
Bonus: $18,000

## Amanda Lewis

2022 Performance
Performance rating: 4.2/5

2023 Performance
Performance rating: 4.5/5

2023 Bonus
Bonus: $14,000
"""
    },

    # Relevant evidence separated by one unrelated section
    {
        "query": "What certification did David Kim obtain in 2023?",
        "document": """
# David Kim

Career
Joined the company in 2021.

2023 Performance
Performance rating: 4.5/5

Projects
Led the Kubernetes migration.

Recognition
Received the Infrastructure Excellence Award.

2023 Certifications
Obtained AWS Solutions Architect Associate certification.

Compensation
2023 bonus: $12,000.
"""
    },
]


for i, test in enumerate(test_cases, start=1):

    print("\n" + "=" * 70)
    print(f"TEST {i}")
    print("=" * 70)

    print("\nQUERY:")
    print(test["query"])

    print("\nCOMPACTED EVIDENCE:")
    result = compact_evidence(
        query=test["query"],
        document=test["document"],
    )

    print(result)