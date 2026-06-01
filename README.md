#  DBMS Normalization Engine

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive, algorithmic engine that automates the mathematical process of database normalization. Built with Python and Streamlit, this application parses raw relational schemas and functional dependencies, mathematically discovers candidate keys, and performs step-by-step recursive decomposition through 1NF, 2NF, 3NF, and BCNF.

##  Key Features

- **Universal Schema Parsing:** The engine is not hardcoded to a specific database. It dynamically parses and evaluates *any* custom relation schema and functional dependency set.
- **Algorithmic Key Discovery:** Automatically calculates attribute closures and uses combinatorics to mathematically prove and extract all valid candidate and primary keys.
- **Minimal Cover Synthesis:** Cleans up functional dependencies by stripping out extraneous attributes and redundant dependencies before decomposition.
- **Step-by-Step Visualization:** Breaks down the logic exactly how a database engineer would approach it, highlighting partial and transitive dependencies before splitting tables.
- **Final BCNF Generation:** Outputs a clean, fully normalized database structure complete with Primary Key (PK) and inferred Foreign Key (FK) join hints.

##  How the Engine Works Under the Hood

Unlike simple rule-based scripts, this engine utilizes pure relational algebra:
1. **1NF:** Validates for atomic attributes and flags potential repeating groups.
2. **2NF:** Detects partial dependencies against calculated candidate keys and isolates them.
3. **3NF:** Synthesizes a new schema utilizing the minimal cover to eliminate transitive dependencies.
4. **BCNF:** Applies a recursive decomposition algorithm to ensure every determinant in the final schema is a valid superkey.

##  Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/yourusername/dbms-normalization-engine.git](https://github.com/yourusername/dbms-normalization-engine.git)
cd dbms-normalization-engine
```
```bash
# Install the required dependencies (Streamlit and core utilities)
pip install streamlit
```

```bash
# Run the Streamlit web application
streamlit run app.py
```

##  Usage Guide

1. **Enter your Relation Schema** in the provided input box (e.g., `R(FlightNo, PilotID, PilotName, GateNo)`).
2. **Enter your Functional Dependencies**, strictly one per line (e.g., `FlightNo -> PilotID, GateNo`).
3. **Click Start Normalization Engine**.
4. **Navigate through the interactive tabs** to explore your Attribute Closures, Candidate Keys, and the step-by-step decomposition of your schema.

##  License

This project is open-source and available under the [MIT License](LICENSE).
