# Real-Time Financial Fraud Detection Using Graph Neural Networks and Anomaly Detection

**A plain-language technical explanation of the published research, its implementation, and its economic significance to the United States.**

## Citation
> Zakaria, R. M., Rahman, M. M., Choudhury, M. T. H., Rahman, H., Rafi, M. A., Minto, A., **Hossain, M. S.**, & Saimon, S. I. (2025). Detecting Financial Fraud in Real-Time Transactions Using Graph Neural Networks and Anomaly Detection Techniques. *Journal of Economics, Finance and Accounting Studies*, 7(6), 1–13.
>
> **DOI:** [10.32996/jefas.2025.7.6.1](https://doi.org/10.32996/jefas.2025.7.6.1)
> **Published:** September 29, 2025 · **License:** CC BY-NC 4.0 · **Peer-reviewed research article**

**As a Computer Science background and Artificial Intelligence and Machine Learning Researcher I have designed and implemented the temporal graph construction module and the streaming anomaly-detection pipeline; conducted the drift and latency experiments reported in Section X."]*

## 1. What this document is

This README explains, in ordinary language, a fraud-detection system published in a peer-reviewed journal. It is written so that a reader without a computer science background can follow **what the problem is, how the system works, why it works better than what banks use today, and what it is worth to the American economy.**

Technical detail is included, but every technical section is preceded by a plain-language explanation of the same idea.

---

## 2. The problem in one paragraph

When you tap a card, the bank has roughly **one-tenth of one second** to decide whether to approve the payment. In that fraction of a second it must judge whether the transaction is legitimate or criminal. Traditional systems make that judgment by looking at the transaction *by itself* — the amount, the location, the time of day. But modern fraud is rarely a single suspicious transaction. It is a **coordinated network**: one stolen device used across forty accounts, one merchant account laundering money for a ring of "customers," hundreds of small transfers that look ordinary alone and only reveal themselves as a pattern when viewed together. A system that examines transactions one at a time is structurally blind to the thing it is trying to catch.

**This research solves that blindness while still answering in under a tenth of a second.**

---

## 3. Why the current defenses fall short

| Approach in wide use today | How it works | Where it fails |
|---|---|---|
| **Rule-based engines** | Human analysts write rules: "flag any transfer over $10,000 to a new payee." | Criminals learn the thresholds and structure their activity to stay just underneath. Rules cannot anticipate a scheme nobody has written a rule for. |
| **Tabular machine learning** (gradient boosting, logistic regression) | Learns from a spreadsheet of transaction attributes. | Each transaction is a row in isolation. The model cannot see that eight "unrelated" rows share one device fingerprint. |
| **Static graph analysis** | Builds a network diagram of accounts, then analyzes it. | The diagram is rebuilt overnight in a batch job. By the time it is refreshed, a fast-moving fraud campaign has already cashed out and disappeared. |

Each of these is a real defense doing real work. The gap this research fills is the space between them: **the relationship-awareness of graph analysis, delivered at the speed of a live payment authorization.**

---

## 4. The core insight: fraud is a network, not a transaction

The system stops treating the payment stream as a list of events and starts treating it as a **living map of relationships**.

Every entity in the payment ecosystem becomes a point on the map (a **node**):

- **Cards / accounts** — who is paying
- **Devices** — the phone, laptop, or terminal used
- **Merchants** — who is being paid
- **IP addresses** — the network location the payment came from

Every transaction becomes a **connection** between those points (an **edge**), stamped with the time it occurred and its attributes (amount, currency, channel).

**A useful analogy for a non-technical reader:** think of an epidemiologist tracing a disease outbreak. Looking at one patient's chart tells you very little. Looking at the *contact map* — who was in the same room as whom, and when — reveals the outbreak. This system builds the contact map of a payment network, and keeps it current to the second.

The critical property is that the map is **temporal** and **heterogeneous**:

- **Temporal** — the map does not just record *that* a card and a device are connected, but *when*, and in what order. A device that touched 40 cards over four years is a shared family tablet. A device that touched 40 cards in eleven minutes is a fraud operation. Only a time-aware map can tell those two apart.
- **Heterogeneous** — the map holds several different *kinds* of entity and several different *kinds* of relationship at once, rather than flattening everything into one undifferentiated network.

---

## 5. How the system works — the mechanism, step by step

Below is the full life of a single transaction moving through the system. The whole sequence completes in **under 100 milliseconds**.

### Step 1 — The transaction arrives and updates the map

A payment request enters the stream. The system immediately inserts it into the temporal graph: it links the card node to the merchant node, attaches the device and IP nodes, and time-stamps every one of those connections. The map is now current as of this instant.

### Step 2 — The neighborhood is gathered

Rather than analyzing the whole map (which would be far too slow), the system pulls only the **local neighborhood** around this transaction: the card's recent counterparties, the device's other recent cards, the merchant's recent customers, and the connections one or two hops further out. This is the difference between reading the entire phone book and reading one person's recent call log.

### Step 3 — The Graph Neural Network produces a "behavioral fingerprint"

This is the technical heart of the system, and it is simpler than its name suggests.

A **Graph Neural Network (GNN)** is a model that describes a point on the map by summarizing the points around it. It repeats a single operation: *look at my neighbors, summarize what they look like, and fold that summary into my own description.* Run that operation twice and each node's description now encodes not only its own history but the character of its entire local neighborhood.

The output is a compact numerical summary — an **embedding** — that functions as a **behavioral fingerprint**. Two accounts with no direct connection to each other, but which both sit inside the same suspicious cluster of shared devices and shell merchants, receive *mathematically similar fingerprints*. **The system can therefore recognize a fraud ring member it has never seen before, purely from the company it keeps.** This is the capability that no transaction-by-transaction model possesses.

The specific architectures used are **GraphSAGE** and **Graph Attention Network (GAT)** variants, chosen because they are *inductive*: they can generate a fingerprint for an account created five seconds ago, without retraining. Fraud rings open new accounts constantly, so a model that must be retrained to recognize a new account is useless in production. Edge features (transaction attributes) and time encoding are incorporated directly, so recency and sequence carry weight.

### Step 4 — Two detectors examine the fingerprint in parallel

The fingerprint is then judged twice, simultaneously, by two different kinds of detector. This dual design is deliberate and is one of the framework's main contributions.

**Detector A — the supervised, cost-sensitive classifier.**
This detector has been trained on confirmed historical fraud. It answers: *does this resemble fraud we have caught before?* It is **cost-sensitive**, meaning it is explicitly taught that the two possible mistakes are not equally expensive. Letting a $30,000 fraudulent wire through costs vastly more than wrongly holding a $40 grocery purchase for two seconds. The model optimizes for *dollars protected*, not for raw accuracy — a distinction with direct financial consequences.

**Detector B — the unsupervised anomaly detectors (Isolation Forest, Deep SVDD).**
This detector has been trained only on what *normal* looks like. It answers a completely different question: *is this simply strange?* It requires no examples of fraud at all.

**Why both are necessary:** Detector A cannot catch an attack pattern that has never been labeled — and every novel fraud scheme is, by definition, unlabeled on the day it launches. Detector B covers exactly that gap. Together they cover both **known fraud** (caught precisely) and **novel, label-sparse fraud** (caught early, before anyone has had time to label it). In practice this is the difference between catching a new scheme on day one versus day ninety.

### Step 5 — The decision is made against an adaptive threshold

The two scores are combined into a decision: approve, decline, or route to a human analyst.

The threshold is not a fixed number. It is tuned on **precision@k** — a measure of how many of the top *k* flagged transactions are genuinely fraudulent. This is a deliberately practical choice: a fraud team can review perhaps 500 alerts per day, so the system is tuned to make **those specific 500 alerts as valuable as possible**, rather than to maximize an abstract statistic. The threshold adapts as conditions change.

### Step 6 — The system explains itself

Whenever a transaction is flagged, the system produces a **subgraph rationale** — via GNNExplainer and motif scoring — identifying *which relationships in the map drove the decision*. In practice the analyst sees something like: *"Flagged because this device connects to 23 cards first used within the last 6 hours, 19 of which paid the same newly-registered merchant."*

**Why this matters far beyond convenience:** U.S. financial institutions operate under fair-lending, adverse-action, and model-risk-management obligations (including the Federal Reserve/OCC SR 11-7 supervisory guidance on model risk). A model that cannot articulate why it declined a customer is a regulatory liability regardless of how accurate it is. **Explainability is what makes this system deployable in a regulated American bank rather than merely publishable.**

### Step 7 — The system keeps learning

Fraud tactics change continuously. This phenomenon — the statistical world shifting underneath a trained model — is called **concept drift**, and it silently degrades every static model in production.

The framework addresses it directly:

- **Drift triggers** — statistical monitors detect when incoming traffic no longer resembles the training distribution and signal for adaptation.
- **Continual learning with replay** — the model updates on new patterns while replaying archived examples, so it learns the new attack *without forgetting* the old one (avoiding the well-known problem of catastrophic forgetting).
- **Streaming reweighting** — the extreme rarity of fraud (often well under 1% of transactions) is corrected continuously in the live stream, rather than only once at training time.

---

## 6. Architecture at a glance

```
   Live payment stream
           │
           ▼
  ┌────────────────────┐
  │ Temporal graph     │   card ── device ── IP
  │ update             │     │        │
  │ (multi-relational) │     └── merchant
  └────────┬───────────┘
           │
           ▼
  ┌────────────────────┐
  │ Lightweight GNN    │   GraphSAGE / GAT
  │ (edge features +   │   → behavioral
  │  time encoding)    │     fingerprint
  └────────┬───────────┘
           │
      ┌────┴────┐
      ▼         ▼
 ┌─────────┐ ┌────────────────┐
 │ Cost-   │ │ Unsupervised   │
 │sensitive│ │ detectors      │
 │classifier│ │ (Isolation    │
 │(known   │ │  Forest,       │
 │ fraud)  │ │  Deep SVDD)    │
 └────┬────┘ └───────┬────────┘
      └──────┬───────┘
             ▼
   ┌──────────────────────┐
   │ Adaptive threshold   │  tuned on precision@k
   │ → approve / decline  │
   │   / analyst review   │
   └──────────┬───────────┘
              │
              ├──► Subgraph rationale (GNNExplainer / motifs)
              │      → analyst + regulatory audit trail
              │
              └──► Drift trigger → continual learning (replay)

  Serving layer: feature cache · micro-batching · asynchronous inference
  Decision budget: < 50–100 ms
```

### The engineering that makes it fast enough to be real

A design that is accurate but too slow is not a fraud system; it is a research demo. The paper specifies a concrete deployment blueprint:

- **Feature cache** — precomputed neighborhood summaries are held in fast memory, so the system is not querying a database during the authorization window.
- **Micro-batching** — transactions arriving within the same few milliseconds are processed as a small group, amortizing computation across them.
- **Asynchronous inference** — graph maintenance is decoupled from the scoring path, so map upkeep never blocks a live decision.
- **Lightweight GNN variants** — model capacity is deliberately constrained to fit the latency budget.

Result: a **sub-50–100 millisecond decision budget**, which is inside the authorization window used by real card networks. This is what moves the work from theory to something a payments processor could deploy.

---

## 7. A worked example: catching a mule network

*This example illustrates the mechanism concretely.*

A criminal group recruits 30 people to open ordinary checking accounts. Stolen funds are moved through these accounts in amounts of $2,000–$4,000 and consolidated offshore.

**What a rule-based system sees:** thirty separate customers making mid-size transfers. Every single transfer is below every alerting threshold. Nothing fires.

**What a tabular ML model sees:** thirty unremarkable rows. The features — amount, hour, merchant category — are within normal ranges for each account individually. Low risk scores across the board.

**What this system sees:** the graph shows that 22 of the 30 accounts were accessed from four shared devices; that all 30 were opened within a 9-day window; that the fund flows converge on two destination nodes; and that this convergence pattern formed in the last 72 hours.

- The **GNN** assigns all 30 accounts nearly identical behavioral fingerprints, because their neighborhoods are structurally identical.
- The **unsupervised detector** flags the cluster as anomalous even though this specific scheme has never been labeled in training data.
- The **explainer** hands the analyst the shared-device subgraph and the convergence pattern.
- The analyst freezes the ring — **as one investigation, not thirty unconnected alerts.**

That is the practical difference, stated without exaggeration: **the same evidence, visible instead of invisible.**

---

## 8. Evaluation

The framework was evaluated on **mixed synthetic and industry datasets containing evolving fraud scenarios**, deliberately constructed so that attack patterns *change over the evaluation period* — testing whether the system holds up as adversaries adapt, not merely on a frozen snapshot.

Metrics were selected for operational meaning rather than headline appeal:

| Metric | What it measures | Why it was chosen |
|---|---|---|
| **ROC-AUC** | Overall ability to separate fraud from legitimate activity | Standard comparability benchmark |
| **PR-AUC** | Performance concentrated on the rare positive class | The honest metric when fraud is <1% of traffic; ROC-AUC alone can flatter a weak model on imbalanced data |
| **Detection delay** | How quickly a new campaign is caught | Every hour of delay is money irrecoverably gone |
| **Alert volume** | Number of cases sent to human review | Determines whether a real fraud team can actually operate the system |
| **Business impact under cost constraints** | Net dollars protected after review costs | The measure an executive or regulator actually cares about |

**Result:** the framework delivered consistent gains over rule-based systems, tabular machine learning, and static graph baselines. The margin was **largest precisely where the need is greatest** — on *low-footprint fraud* (schemes deliberately structured to stay under thresholds) and *fast-moving attack campaigns* (schemes that begin and end before a batch system refreshes).

> **⚠ ACTION REQUIRED BEFORE FILING:** Replace this box with the exact figures from the published tables — the reported ROC-AUC, PR-AUC, detection delay, and baseline comparison numbers. A USCIS officer weighs *specific verifiable numbers* far more heavily than qualitative claims, and every figure in the petition must match the published paper exactly.

---

## 9. Why this matters to the United States economy

The mechanism above is not an abstract contribution. It targets a documented and rapidly worsening national problem, using an approach the U.S. federal government has already validated on its own payment systems.

### 9.1 The scale of the problem is federally documented and accelerating

- **Federal Trade Commission:** <cite index="13-6">Americans reported losing about $16 billion to fraud in 2025 — the highest figure on record and roughly 25% higher than 2024.</cite> <cite index="16-1">Reported losses have risen nearly 430% since 2020.</cite>
- **Federal Bureau of Investigation (IC3):** <cite index="24-1">Losses reported to the Internet Crime Complaint Center reached $20.9 billion in 2025, a 26% increase over the prior year, across 1,008,597 complaints — the first time annual complaints exceeded one million in the center's 25-year history.</cite>
- <cite index="21-1">Cyber-enabled fraud accounts for roughly 85% of all reported losses</cite>, and <cite index="21-1">the FBI's 2025 report documents growing criminal use of artificial intelligence, with more than 22,000 complaints and nearly $900 million in associated losses.</cite>
- **Who bears the loss:** <cite index="14-1">Americans aged 50 and older reported $4.3 billion lost to fraud in 2025, compared with $2.3 billion among younger adults.</cite>

**The trend line is the argument.** Losses are not merely large; they are compounding at 25–26% annually while criminals adopt AI faster than defenses adapt. Static, rule-based defenses are losing ground every year. This research addresses the specific structural reason why.

### 9.2 The U.S. government has already proven this exact approach works — at scale, with taxpayer money

This is the strongest available evidence that the endeavor has national importance, because the federal government has independently validated the method:

- <cite index="8-1">The U.S. Department of the Treasury announced that its technology- and data-driven approach to fraud and improper payment prevention — including machine learning AI — enabled the prevention and recovery of over $4 billion in fraud and improper payments in fiscal year 2024, up from $652.7 million in FY2023.</cite>
- The FY2024 breakdown: <cite index="5-1">$2.5 billion from identifying and prioritizing high-risk transactions, $1 billion recovered by expediting check-fraud identification with machine learning, and $500 million from expanded risk-based screening.</cite>
- A senior Treasury official <cite index="7-1">described the results as transformative, noting that leveraging data had substantially improved the department's fraud detection and prevention capability.</cite>
- Scale of exposure: <cite index="6-1">Treasury manages roughly 1.4 billion transactions worth approximately $1.7 trillion annually.</cite>

**The direct connection:** Treasury achieved a six-fold improvement using machine learning that analyzes transactions largely in isolation. **This research advances the frontier one full step further** — adding *relationship-aware* detection (catching coordinated rings, not just individual anomalies), *real-time* operation (sub-100ms rather than batch), and *built-in explainability* (satisfying audit and regulatory review). It is a direct technical contribution to a capability the United States government has publicly committed to and is actively investing in.

### 9.3 Where the economic benefit lands

| Beneficiary | Concrete effect |
|---|---|
| **Consumers and retirees** | Earlier interception means funds are frozen while still recoverable. Detection delay is measured in the paper precisely because recovery probability collapses within hours. |
| **Banks and payment processors** | Fewer losses absorbed, and materially fewer *false declines*. False declines are a large, under-discussed cost: legitimate customers blocked at checkout abandon purchases and often leave the institution entirely. Precision@k tuning attacks this cost directly. |
| **Merchants and small businesses** | Chargeback losses and payment friction fall. Small merchants operating on thin margins are disproportionately harmed by both fraud and over-aggressive blocking. |
| **Federal and state programs** | The same architecture transfers directly to public-benefit program integrity — Medicaid, Medicare, unemployment insurance — where coordinated billing and provider-collusion fraud is *inherently* a network problem that transaction-level review cannot see. |
| **Regulators and supervised institutions** | The explainability layer produces the audit trail that model-risk-management supervision requires, lowering the compliance barrier to deploying advanced detection at all. |
| **U.S. technological competitiveness** | Applied graph machine learning in financial infrastructure is a strategically contested field. Domestic research capacity and domestic talent retention in this area are matters of national economic security. |

### 9.4 A note on honest framing

This system does not eliminate fraud, and this document does not claim it does. What it does is close a specific, identifiable, and expensive gap: **coordinated fraud rings that are individually invisible and collectively devastating, operating faster than batch-refreshed defenses can respond.** Given the federally documented figures above, even a modest proportional improvement against that specific category represents hundreds of millions of dollars annually retained in the U.S. economy — and the government's own six-fold FY2024 result demonstrates that improvements in this domain are not incremental in practice.

---

## 10. From publication to U.S. deployment

*[Customize this section — it is the strongest part of the argument, because it demonstrates that the research is not confined to a journal.]*

The methods in this paper are directly continuous with my current professional and research work in the United States:

- **Applied practice:** As a Software Engineer at 1st Aide Home Care Inc. (New York), I build billing-integrity and compliance-verification systems for Medicaid-funded home care — including Medicaid billing automation, Electronic Visit Verification (EVV) compliance tracking, and multi-payer authorization workflows. Medicaid billing fraud is structurally identical to the payment-network problem described above: coordinated actors, individually plausible claims, patterns visible only in the relationships between providers, aides, patients, and visit records.

- **Continuing research:** *"Safeguarding Public Healthcare Spending with Explainable AI: A Statistically Validated Machine Learning Framework for Medicaid Fraud Detection and Electronic Visit Verification,"* Frontiers in Computer Science and Artificial Intelligence, 5(3), 35–44 (2026). DOI: 10.32996/jcsts.2026.5.3.4. This work applies the same explainable-anomaly-detection philosophy to a real CMS dataset protecting U.S. public healthcare spending.

- **Trajectory:** the through-line across this record is a single national endeavor — **AI-driven, explainable, real-time fraud and anomaly detection protecting U.S. financial and public-benefit systems.** The published financial-fraud research establishes the methodological foundation; the Medicaid work applies it to federal program integrity; the professional role operationalizes it inside a U.S. healthcare provider.

---

## 11. Reproducibility and technical stack

*[Complete this section only if you are publishing an accompanying code repository. If you are not, delete it — an empty or aspirational section weakens the document.]*

```
├── data/                  # dataset loaders, synthetic scenario generators
├── graph/                 # temporal multi-relational graph construction & updates
├── models/
│   ├── gnn/               # GraphSAGE / GAT with edge features + time encoding
│   ├── supervised/        # cost-sensitive classifier
│   └── unsupervised/      # Isolation Forest, Deep SVDD
├── streaming/             # reweighting, drift triggers, continual learning (replay)
├── explain/               # GNNExplainer, motif scoring, subgraph rationales
├── serving/               # feature cache, micro-batching, async inference
└── eval/                  # ROC-AUC, PR-AUC, detection delay, alert volume, cost model
```

**Core stack:** Python · PyTorch · PyTorch Geometric / DGL · scikit-learn · streaming ingestion layer (e.g. Kafka) · in-memory feature store (e.g. Redis)

---

## 12. Glossary for the non-technical reader

| Term | Plain meaning |
|---|---|
| **Graph** | A map of things and the connections between them |
| **Node / Edge** | A point on the map / a connection between two points |
| **Graph Neural Network (GNN)** | A model that describes each point by summarizing its surroundings |
| **Embedding** | A compact numerical "fingerprint" of an entity's behavior and context |
| **Temporal / dynamic graph** | A map that records *when* each connection formed and updates continuously |
| **Heterogeneous graph** | A map holding several different kinds of entity and relationship at once |
| **Anomaly detection** | Learning what normal looks like, then flagging departures from it |
| **Class imbalance** | The problem that genuine fraud is a tiny fraction of all transactions |
| **Concept drift** | The problem that fraud tactics change after a model is trained |
| **Cost-sensitive learning** | Teaching the model that some mistakes are far more expensive than others |
| **Precision@k** | Of the top *k* alerts sent to human analysts, how many are real |
| **Latency** | How long the system takes to answer — here, under 100 milliseconds |
| **Explainability (XAI)** | The system's ability to state *why* it made a given decision |

---

## 13. Sources

1. Zakaria, R. M., et al. (2025). *Detecting Financial Fraud in Real-Time Transactions Using Graph Neural Networks and Anomaly Detection Techniques.* Journal of Economics, Finance and Accounting Studies, 7(6), 1–13. https://doi.org/10.32996/jefas.2025.7.6.1
2. U.S. Department of the Treasury. *Treasury Announces Enhanced Fraud Detection Processes, Including Machine Learning AI, Prevented and Recovered Over $4 Billion in Fiscal Year 2024.* Press Release JY2650, October 17, 2024. https://home.treasury.gov/news/press-releases/jy2650
3. U.S. Department of the Treasury. *Treasury Announces Enhanced Fraud Detection Process Using AI Recovers $375M in Fiscal Year 2023.* Press Release JY2134. https://home.treasury.gov/news/press-releases/jy2134
4. Federal Trade Commission. *FTC Data Show People Reported Losing $3.5 Billion to Imposter Scams in 2025.* June 2026. https://www.ftc.gov/news-events/news/press-releases/2026/06/ftc-data-show-people-reported-losing-3-point-5-billion-imposter-scams-2025
5. Federal Bureau of Investigation, Internet Crime Complaint Center. *2025 Internet Crime Report.* April 2026. https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf
6. Hossain, M. S., & Hasan, A. T. M. F. (2026). *Safeguarding Public Healthcare Spending with Explainable AI.* Frontiers in Computer Science and Artificial Intelligence, 5(3), 35–44. https://doi.org/10.32996/jcsts.2026.5.3.4
