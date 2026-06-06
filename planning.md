# Project 1 Planning: The Unofficial Guide

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
| 3 | Stanford | Social Security Number for students | https://bechtel.stanford.edu/navigate-international-life/social-security-number-ssn |
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

**Chunk size: 500 tokens**

**Overlap: 100 tokens**

**Chunking Strategy: Semantic chunking**

**Reasoning:**
Documents are FAQs, university guides, and government resources containing multiple paragraph of explanations that need more context. Smaller chunks could separate related information. A 100 token overlap ensures that information near chunk boundaries appears in both neighboring chunks. Since my documents consists of educational guides and government resources, I would prefer paragraph based chunking with token limits, because preserving the structure improves retrieval quality. A 200 character chunk might contain only part of the explanation, causing retrieval to return incomplete information and reducing answer quality, hence I'm using 500 character chunk limit.

---

## Retrieval Approach

**Embedding model: all-MiniLM-L6-v2 Sentence-Transformers**

**Top-k: 4 to 6 chunks**

**Production tradeoff reflection:**
- `all-MiniLM-L6-v2` is chosen because the documents are a combination of formal and informal texts. So we need embeddings that can handle paraphrases, mix of formal and informal text and long explanatory passages. Semantic search works best here because embeddings capture meaning, not keywords. 

- 4 to 6 chunks should be sufficient to to capture definition, requirements, steps and process details. More thank 7 chunks could introduce noise and irrelevant context and 1 or 2 chunks is too narrow and might miss the context. 

- Embedding model can be upgraded to `bge-large-en` for production because it offers better accuracy and reduce hallucination risk. They are generally slow, but that is acceptable for the quality of results. It can further be extended to support multilingual queries for international students. 

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Can an international student on an F-1 visa work off-campus during their first academic year? | No, first-year students are limited to on-campus employment. Off-campus work requires authorization: CPT, OPT, Rare hardship authorization |

| 2 | Do international students in the U.S. need to file taxes even if they have no income? | Yes. Must file Form 8843 even with zero income. If employed, may also file 1040-NR. Requirement is based on visa status (F-1), not income level alone |

| 3 | How can an international student rent an apartment in the U.S. without a credit history? | Use a U.S. guarantor or co-signer. Pay a larger security deposit. Provide bank statements or proof of funds. Show scholarship or income proof if available. Use university housing or student housing options. Use guarantor services (paid options). Provide identity and visa documents (passport, I-20, F-1 visa) |

| 4 | What should an international student do if they feel unsafe on campus at night in the U.S.? | Use campus escort / safety walk services. Call campus police or 911 in emergencies. Stay in well-lit, populated areas. Use university safety apps or emergency phones. Avoid isolated routes and inform a friend if possible |

| 5 | What are common signs of culture shock for international students in the U.S., and how do students typically cope with it? | Symptoms: Homesickness, Isolation or loneliness, Frustration with communication/culture differences, Fatigue or anxiety in social situations
Coping: Join student clubs or international student groups, Talk to counseling or international student services, Build small routines (gym, study spots, cafes), Give time for adjustment (normal phases) |

| 6 | How does an international student open a bank account in the U.S., and what documents are usually required? | Go to a bank or credit union (or sometimes online onboarding is allowed). Required documents typically include: Passport, I-20 form, F-1 visa, Proof of U.S. address (lease, utility bill, or school housing letter), Sometimes SSN or ITIN (if available, but often not required initially). Student may start with a checking account. Some banks offer student-friendly accounts with no minimum balance. Debit card is usually issued after account setup |

---

## Anticipated Challenges

1. Loss of context at chunk boundaries: A single important concept like CPT rules or OPT deadlines is split across two chunks. One chunk might contain eligibility rules, while the other contains timing or exceptions.

2. Mix of low and high quality sources

3. Incorrect retrieval: Semantic search returns chunks that are related in meaning but not actually answering the question.

4. Lack of source traceability: Generating answers without preserving which source each fact came from.

---

## Architecture

flowchart LR

A[Document Ingestion<br/>Sources: USCIS, IRS, University Pages, Reddit] 
--> B[Chunking<br/>Tool: Python <br/>500 tokens + 100 overlap]

B --> C[Embedding + Vector Store<br/>Model: all-MiniLM-L6-v2<br/>Store: ChromaDB]

C --> D[Retrieval<br/>Top-k: 4–6 chunks<br/>Semantic Search]

D --> E[Generation<br/>Prompt + context = Answer]

## AI Tool Plan

 - AI Tool: ChatGPT
 - Input: planning.md - Documents section
 - Expected Output: Clean text results
 - Output Verification: Manually check 3 to 5 pages for missing sections, junk text (menus, footers) and valid text.

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
