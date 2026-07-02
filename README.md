# Decision Guardian
Decision Guardian is a lightweight CI tool that connects **architecture decisions (ADRs)** to code changes in **GitLab Merge Requests**.
It automatically detects when changed files match architectural rules defined in markdown and surfaces them directly in MR reviews.
---
## 🚀 What it does
- Reads architecture decisions from markdown files
- Detects files changed in a GitLab Merge Request
- Matches changes against decision rules (glob patterns)
- Posts a summary comment on the MR
- Optionally fails CI on critical violations
---
## 🧭 How it works
```mermaid
flowchart TD
    A[Merge Request Opened] --> B[GitLab CI Triggered]
    B --> C[Fetch Changed Files]
    C --> D[Load /decisions/*.md]
    D --> E[Parse Decisions into Rules]
    E --> F[Match Rules vs Changed Files]
    F --> G[Generate Violation Report]
    G --> H[Post MR Comment]
    G --> I{Critical Violations?}
    I -->|Yes| J[Fail CI Pipeline]
    I -->|No| K[Pass Pipeline]

⸻

📁 Decision Format

Store decisions in:

/decisions

Example:

# Database Connection Rule
Severity: critical
Applies to:
- src/db/**
Reason:
Only one database connection pool should exist.

⸻

⚙️ Installation (GitLab CI)

Add this job to your .gitlab-ci.yml:

decision_guardian:
  image: node:20
  script:
    - npm install
    - npm run build
    - node dist/index.js check

⸻

🔧 Configuration

Create a .decision-guardian.yml file:

decisions_path: ./decisions
fail_on: critical
comment: true

⸻

🧪 Output Example

MR Comment

⚠ Decision Guardian Report
Critical:
- Database Connection Rule → src/db/pool.ts
Warnings:
- Auth Rule → src/auth/session.ts

⸻

❌ CI Failure Example

If critical violations are found:

❌ Decision Guardian failed: critical architectural violation detected

⸻

🧠 Why use this?

Most architecture decisions live in docs that nobody reads.

Decision Guardian brings them into the merge request workflow, where they actually affect engineering decisions.

⸻

🛠 MVP Scope

Included

* Markdown-based decisions
* Glob-based file matching
* GitLab CI integration
* MR comment bot
* CI failure on critical rules

Not included (yet)

* Regex / AST rules
* AI-based reasoning
* UI dashboard
* Cross-repo dependency tracking
* IDE plugin

⸻

📌 Example Workflow

sequenceDiagram
    participant Dev as Developer
    participant GitLab as GitLab MR
    participant CI as CI Pipeline
    participant DG as Decision Guardian
    Dev->>GitLab: Open Merge Request
    GitLab->>CI: Trigger pipeline
    CI->>DG: Run check command
    DG->>GitLab: Fetch changed files
    DG->>DG: Load + parse decisions
    DG->>DG: Match rules
    DG->>GitLab: Post MR comment
    alt Critical violation
        DG->>CI: Exit(1)
    else No critical issues
        DG->>CI: Exit(0)
    end

⸻

📄 License

MIT