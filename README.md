# The Unofficial Guide — Project 1

## Domain

**International Student Survival Guide (United States)**

Millions of international students come to the United States each year and must navigate immigration rules, academics, housing, healthcare, banking, taxes, employment, and daily life. Information exists across many websites, government agencies, university pages, and online communities, making it difficult to find answers quickly.

International students frequently have practical questions such as:

* What is CPT or OPT?
* How do I rent an apartment without a credit history?
* How do U.S. taxes work for international students?
* How do I open a bank account and build credit?
* What should I do in a medical emergency?
* How can I prepare for internships and jobs?

This knowledge is valuable because students often need information that combines official rules with practical advice. Official sources cover only one aspect of the student experience, while real-world guidance is scattered across universities, government agencies, and student communities.

---

## Document Sources

| #  | Source                               | Type            | URL or file path                                                                                                        |
| -- | ------------------------------------ | --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 1  | DHS                                  | Government      | https://studyinthestates.dhs.gov/students                                                                               |
| 2  | USCIS                                | Government      | https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors                                       |
| 3  | Stanford                             | University      | https://bechtel.stanford.edu/navigate-international-life/social-security-number-ssn                                     |
| 4  | IRS                                  | Government      | https://www.irs.gov/individuals/international-taxpayers                                                                 |
| 5  | Healthcare.gov                       | Government      | https://www.healthcare.gov/young-adults/college-students/                                                               |
| 6  | Consumer Financial Protection Bureau | Government      | https://www.consumerfinance.gov/consumer-tools/bank-accounts/                                                           |
| 7  | MyFICO                               | Educational     | https://www.myfico.com/credit-education                                                                                 |
| 8  | FTC                                  | Government      | https://consumer.ftc.gov/articles/tenant-background-checks-and-your-rights                                              |
| 9  | Ready.gov                            | Government      | https://www.ready.gov/campus                                                                                            |
| 10 | r/IntltoUSA                          | Community Forum | https://www.reddit.com/r/IntltoUSA/                                                                                     |
| 11 | UC Berkeley Career Center            | University      | https://career.berkeley.edu/communities/international-students/                                                         |
| 12 | University of Illinois               | University      | https://provost.illinois.edu/policies/policies/academic-integrity/students-quick-reference-guide-to-academic-integrity/ |
| 13 | Stanford University                  | University      | https://bechtel.stanford.edu/navigate-international-life/taxes                                                          |
| 14 | USC Office of International Services | University      | https://ois.usc.edu/2024/04/08/navigating-culture-shock-a-guide-for-international-students-in-the-u-s/                  |

---

## Chunking Strategy

**Chunk size:** 500 tokens

**Overlap:** 100 tokens

**Why these choices fit your documents:**

My corpus consists primarily of government resources, university guides, FAQs, and community discussions. These documents often contain multi-paragraph explanations, eligibility requirements, and step-by-step procedures. Smaller chunks could separate related information and reduce retrieval quality.

I use semantic chunking with paragraph and section boundaries whenever possible. A 100-token overlap helps preserve context when important information appears near chunk boundaries. This reduces the risk of incomplete retrieval when concepts such as CPT eligibility or tax filing requirements span multiple paragraphs.

**Preprocessing performed:**

* Removed HTML tags
* Removed navigation menus and footers
* Extracted article content only
* Normalized whitespace

**Final chunk count:**
60

---

## Embedding Model

**Model used:** all-MiniLM-L6-v2 (Sentence-Transformers)

**Why this model was chosen:**

The corpus contains both formal and informal text, including government regulations, university guidance, and student discussions. The model provides strong semantic similarity performance while remaining lightweight and efficient.

**Production tradeoff reflection:**

If cost were not a constraint, I would consider using a larger embedding model such as bge-large-en. Larger models generally provide better semantic accuracy and can reduce retrieval errors. Additional considerations would include multilingual support for international students, context length limitations, inference latency, and whether embeddings are generated locally or through a hosted API.

---

## Grounded Generation

**System prompt grounding instruction:**

"Answer the user's question using only the retrieved context. If the answer cannot be found in the provided documents, state that the information is unavailable in the retrieved sources. Do not invent policies, requirements, deadlines, or procedures."

**How source attribution is surfaced in the response:**

Each retrieved chunk retains metadata including the source URL and document title. The generated response includes references to the documents used so users can verify the information against the original source material.

---

## Evaluation Report
| # | Question                                                                                      | Expected answer                                                        | System response (summarized)                                                                                                                                 | Retrieval quality  | Response accuracy  |
| - | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ | ------------------ |
| 1 | Can an international student on an F-1 visa work off-campus during their first academic year? | No, first-year students are generally limited to on-campus employment. | Retrieved information about CPT, OPT, and on-campus employment, but did not clearly answer the question and deferred to the international student office.    | Partially relevant | Partially accurate |
| 2 | Do international students in the U.S. need to file taxes even if they have no income?         | Yes, Form 8843 must generally be filed even with no income.            | Retrieved tax-related information but focused on tax residency and filing requirements. Did not explicitly mention Form 8843 or clearly answer the question. | Partially relevant | Partially accurate |
| 3 | How can an international student rent an apartment in the U.S. without a credit history?      | Use a guarantor, proof of funds, larger deposit, student housing, etc. | Suggested co-signers, larger deposits, and alternative rental options. Also discussed tenant background checks and student support resources.                | Relevant           | Accurate           |
| 4 | What should an international student do if they feel unsafe on campus at night in the U.S.?   | Use campus safety resources, escorts, campus police, or 911.           | Retrieved general campus emergency planning information but lacked specific safety guidance. Generated generic safety recommendations instead.               | Partially relevant | Partially accurate |
| 5 | What are common signs of culture shock for international students and how can they cope?      | Homesickness, isolation, joining communities, counseling, routines.    | Correctly identified common symptoms and coping strategies, including counseling, student groups, and support networks.                                      | Relevant           | Accurate           |

**Retrieval quality:** Partially relevant

**Response accuracy:** Partially accurate

---

## Failure Case Analysis

**Question that failed:**
What is CPT?

**What the system returned:**
I don't have enough information in my sources to answer that. Please check with your university's international student office or the relevant government agency.


**Root cause (tied to a specific pipeline stage):**
Potential causes include:

* The chunk containing "Curricular Practical Training (CPT)" never got embedded.
* Some embedding models struggle with acronyms if the acronym appears infrequently.
* Neither chunk alone fully answers the question.
* Then retrieval returns a chunk, but not the perfect definition chunk.

**What you would change to fix it:**
Possible improvements include larger overlap, reranking retrieved chunks, filtering low-quality sources, or improving metadata handling.

---

## Spec Reflection

**One way the spec helped you during implementation:**

The planning document forced me to think about data collection, chunking, retrieval, and evaluation before writing code. This reduced trial-and-error and made it easier to justify design decisions such as chunk size and retrieval depth.

**One way your implementation diverged from the spec, and why:**

The original plan focused primarily on fixed token chunking, but I expanded it to use semantic chunking based on document structure whenever possible. This better preserves the meaning of university guides and government resources.

---

## AI Usage

### Instance 1

* **What I gave the AI:** The Document Sources and Chunking Strategy sections from planning.md.
* **What it produced:** Python code for document extraction, preprocessing, and semantic chunking with configurable token size and overlap.
* **What I changed or overrode:** I modified the chunking implementation to preserve section headings and paragraph boundaries instead of relying solely on fixed token counts.

### Instance 2

* **What I gave the AI:** The Embedding Model and Retrieval sections from planning.md.
* **What it produced:** ChromaDB indexing code using all-MiniLM-L6-v2 embeddings and top-k retrieval.
* **What I changed or overrode:** I adjusted retrieval parameters to return 4–6 chunks and added source metadata so responses could include document attribution.
