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