# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

International Student Survival Guide (United States)

**Why this domain?**

Millions of international students come to the United States each year and must navigate immigration rules, academics, housing, healthcare, banking, taxes, employment, and daily life. Information exists across many websites, government agencies, university pages, and online communities, making it difficult to find answers quickly.

International students frequently have practical questions such as:

* What is CPT or OPT?
* How do I rent an apartment without a credit history?
* How do U.S. taxes work for international students?
* How do I open a bank account and build credit?
* What should I do in a medical emergency?
* How can I prepare for internships and jobs?

Having a single knowledge source that combines these topics can save students significant time and help them avoid costly mistakes.

**Why is it hard to find through official channels?**

Official sources often cover only a small part of the student experience:

* Important information is scattered across multiple agencies such as DHS, USCIS, IRS, SSA, and individual universities.
* Many real-world questions are answered only through student experiences shared on forums and community discussions.
* Students often need information that combines rules and practical advice.

As a result, students must search across dozens of sources and piece together information themselves. An AI-powered survival guide can aggregate authoritative information and community knowledge into one accessible resource.

---

## Documents

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | DHS | Maintaining F-1 status | https://studyinthestates.dhs.gov/students |
| 2 | USCIS | USCIS student employment (OPT/CPT) | https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors |
| 3 | SSA | Social Security Number for students | https://www.ssa.gov/ssnumber |
| 4 | IRS | IRS tax information for international students | https://www.irs.gov/individuals/international-taxpayers |
| 5 | Healthcare.gov | Student health insurance basics | https://www.healthcare.gov/young-adults/college-students/ |
| 6 | ConsumerFinance | U.S. banking basics | https://www.consumerfinance.gov/consumer-tools/bank-accounts/ |
| 7 | MyFico | Credit score education | https://www.myfico.com/credit-education |
| 8 | FTC | Renting apartments guide | https://consumer.ftc.gov/articles/tenant-background-checks-and-your-rights |
| 9 | Ready.gov | Campus safety resources | https://www.ready.gov/campus |
| 10 | Reddit | Reddit discussions from international students | https://www.reddit.com/r/IntltoUSA/ |
| 11 | UC Berkeley | Career services for international students | https://career.berkeley.edu/communities/international-students/ |
| 12 | University of Illinois | Academic integrity/plagiarism | https://provost.illinois.edu/policies/policies/academic-integrity/students-quick-reference-guide-to-academic-integrity/|
| 13 | UC Berkeley | Internship & Career Preparation | https://career.berkeley.edu/communities/international-students/ |
| 14 | Stanford University | Taxes | https://bechtel.stanford.edu/navigate-international-life/taxes |
| 15 | USC | Culture Shock | https://ois.usc.edu/2024/04/08/navigating-culture-shock-a-guide-for-international-students-in-the-u-s/ |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**

**Overlap:**

**Reasoning:**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.

2.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
