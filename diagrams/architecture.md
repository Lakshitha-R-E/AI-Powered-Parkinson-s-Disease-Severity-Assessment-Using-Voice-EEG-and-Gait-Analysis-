# IEEE Paper Diagrams (Mermaid Code)

## 1. System Architecture Diagram

```mermaid
graph TD
    subgraph Data Sources & Input Modalities
        V["Voice Signal (.wav)<br>UCI & mPower"]
        E["EEG Signal (.edf)<br>PRED+CT Dataset"]
        G["Gait/IMU Signal (.csv)<br>PhysioNet Dataset"]
    end

    subgraph Signal Processing & Feature Extraction
        VP["Voice Processor<br>(MFCC, Pitch, Jitter, Shimmer, HNR)"]
        EP["EEG Processor<br>(MNE, Band Power, Entropy, PLV)"]
        GP["Gait Processor<br>(Step/Stride, Cadence, Asymmetry)"]
    end

    subgraph Deep Learning Encoders
        VE["Voice Encoder<br>(1D-CNN + Residual)"]
        EE["EEG Encoder<br>(CNN + Multi-Head Attention)"]
        GE["Gait Encoder<br>(BiLSTM + Temporal Attention)"]
    end

    subgraph Multimodal Transformer Fusion
        TF["Transformer Fusion Network<br>(Cross-Attention & Modality Dropout)"]
    end

    subgraph Dual-Task Prediction Heads
        RH["UPDRS Regression Head<br>(MSE / Huber Loss)"]
        CH["Severity Classification Head<br>(Cross-Entropy Loss)"]
    end

    subgraph Explainable AI & Clinical Report
        XAI["XAI Module<br>(SHAP, Grad-CAM, Attention Viz)"]
        REP["Report Generator<br>(ReportLab Clinical PDF)"]
    end

    V --> VP --> VE --> TF
    E --> EP --> EE --> TF
    G --> GP --> GE --> TF

    TF --> RH & CH
    RH --> OutputUPDRS["UPDRS Score (0-108)"]
    CH --> OutputClass["Severity: Mild / Moderate / Severe"]

    TF & RH & CH --> XAI --> REP
```

## 2. Data Flow Diagram (DFD Level 1)

```mermaid
flowchart LR
    Patient["Patient / Clinician"] -->|Upload Wav/EDF/CSV| Preprocess["1. Signal Preprocessing & Feature Extraction"]
    Preprocess -->|Feature Vectors| Encoders["2. Deep Learning Feature Encoders"]
    Encoders -->|Modal Embeddings| Fusion["3. Transformer Cross-Modal Fusion"]
    Fusion -->|Unified Representation| DualHead["4. Dual-Task Learning Head"]
    DualHead -->|Predictions & Loss| ModelStore["Model Weight Storage"]
    DualHead -->|UPDRS & Class| XAI["5. Explainable AI & SHAP Engine"]
    XAI -->|Attributions & Attention| Report["6. PDF Report Generator"]
    Report -->|Clinical Report PDF| Patient
```

## 3. Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    USERS ||--o{ PATIENTS : "creates"
    USERS ||--o{ PREDICTIONS : "executes"
    USERS ||--o{ REPORTS : "generates"
    PATIENTS ||--o{ PREDICTIONS : "has"
    PATIENTS ||--o{ REPORTS : "has"
    PREDICTIONS ||--o{ REPORTS : "produces"

    USERS {
        uuid id PK
        string username
        string email
        string hashed_password
        string role
    }

    PATIENTS {
        uuid id PK
        string patient_code UK
        string first_name
        string last_name
        date date_of_birth
        string gender
    }

    PREDICTIONS {
        uuid id PK
        uuid patient_id FK
        float updrs_score
        int severity_class
        string severity_label
        float confidence
        jsonb voice_features
        jsonb eeg_features
        jsonb gait_features
        timestamp assessed_at
    }

    REPORTS {
        uuid id PK
        uuid prediction_id FK
        uuid patient_id FK
        string file_path
        string report_type
        timestamp generated_at
    }
```

## 4. Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Clinician
    participant UI as Streamlit App
    participant API as FastAPI Backend
    participant Pipe as Signal Processors
    participant Model as PyTorch Multimodal Model
    participant XAI as SHAP / XAI Engine
    participant PDF as ReportLab Generator

    Clinician->>UI: Upload Voice (.wav), EEG (.edf), Gait (.csv)
    UI->>API: POST /predict (multipart files)
    API->>Pipe: Run feature extraction (Voice, EEG, Gait)
    Pipe-->>API: Extracted numerical feature vectors
    API->>Model: Forward pass (Modality Encoders + Transformer Fusion)
    Model-->>API: UPDRS Score & Severity Class
    API-->>UI: JSON Result (UPDRS, Severity, Confidence)
    UI->>Clinician: Display Dashboard, Radar & Gauge Charts

    Clinician->>UI: Click "Generate Report"
    UI->>XAI: Compute SHAP values & Attention Heatmaps
    XAI-->>UI: Feature importance plots
    UI->>PDF: Generate Clinical PDF Report
    PDF-->>UI: Downloadable PDF File
    UI->>Clinician: PDF Download Link
```

## 5. Use Case Diagram

```mermaid
graph LR
    subgraph Users
        Clinician(("Clinician / Neurologist"))
        Researcher(("AI Researcher"))
        Admin(("System Admin"))
    end

    subgraph System Features
        UC1["Upload Multimodal Signals"]
        UC2["View Signal Processing Plots"]
        UC3["Predict UPDRS & Severity"]
        UC4["Inspect XAI & SHAP Dashboards"]
        UC5["Generate Clinical PDF Report"]
        UC6["Trigger Model Retraining"]
        UC7["Manage Patient Records & History"]
    end

    Clinician --> UC1
    Clinician --> UC2
    Clinician --> UC3
    Clinician --> UC4
    Clinician --> UC5
    Clinician --> UC7

    Researcher --> UC1
    Researcher --> UC3
    Researcher --> UC4
    Researcher --> UC6

    Admin --> UC7
```
