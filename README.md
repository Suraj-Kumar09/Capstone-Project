<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![DagsHub](https://img.shields.io/badge/DagsHub-MLOps-green)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?logo=amazonaws)
![S3](https://img.shields.io/badge/AWS-S3-red)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue?logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-blue?logo=kubernetes)
![EKS](https://img.shields.io/badge/AWS-EKS-orange)
![ECR](https://img.shields.io/badge/AWS-ECR-yellow)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-black?logo=githubactions)
![Flask](https://img.shields.io/badge/Flask-Web%20API-black?logo=flask)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboarding-orange?logo=grafana)

</p>



## 🛠️ Core Tech Stack

🐍 Python • 🤖 Scikit-Learn • 📦 DVC • 📊 MLflow • 🌐 Flask • ☁️ AWS • 🗄️ S3 • 🐳 Docker • ☸️ Kubernetes (EKS) • 📥 ECR • 🔄 GitHub Actions • 📈 Prometheus • 📊 Grafana


# 🚀 Enterprise MLOps Pipeline with AWS, Kubernetes & Observability

### End-to-End Machine Learning Lifecycle Automation using DVC, MLflow, Docker, AWS EKS, GitHub Actions, Prometheus & Grafana

---

## 🌟 Project Overview

This project demonstrates a complete production-grade MLOps workflow that automates the entire machine learning lifecycle — from data ingestion and preprocessing to model training, experiment tracking, deployment, monitoring, and cloud-native scalability.

The solution integrates modern MLOps tools and AWS services to create a reproducible, scalable, and observable ML platform.

### Key Highlights

✅ Automated ML Pipeline using DVC

✅ Experiment Tracking with MLflow + DagsHub

✅ AWS S3-based Data & Artifact Versioning

✅ CI/CD Automation using GitHub Actions

✅ Dockerized Application Deployment

✅ AWS ECR Image Registry

✅ Kubernetes Deployment on AWS EKS

✅ Real-Time Monitoring with Prometheus

✅ Interactive Dashboards with Grafana

✅ Production-Ready Infrastructure

---

# 🏗️ Solution Architecture

```text
                ┌──────────────────┐
                │   Raw Dataset    │
                └────────┬─────────┘
                         │
                         ▼
              ┌───────────────────┐
              │ Data Ingestion    │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ Data Processing   │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ Feature Engineering│
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ Model Training    │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ Model Evaluation  │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │ MLflow Tracking   │
              └────────┬──────────┘
                       │
                       ▼
                 AWS S3 + DVC
                       │
                       ▼
              GitHub Actions CI/CD
                       │
                       ▼
                   Docker
                       │
                       ▼
                    AWS ECR
                       │
                       ▼
                   AWS EKS
                       │
                       ▼
                Flask API Service
                       │
                       ▼
          Prometheus + Grafana
```

---

# 🛠️ Tech Stack

| Category           | Technologies   |
| ------------------ | -------------- |
| Language           | Python         |
| ML Tracking        | MLflow         |
| Experiment Hub     | DagsHub        |
| Data Versioning    | DVC            |
| Storage            | AWS S3         |
| API                | Flask          |
| Containerization   | Docker         |
| CI/CD              | GitHub Actions |
| Container Registry | AWS ECR        |
| Orchestration      | Kubernetes     |
| Cloud Platform     | AWS            |
| Monitoring         | Prometheus     |
| Visualization      | Grafana        |

---

# 📂 Project Structure

```text
CAPSTONE-PROJECT
│
├── 📂 data/                    # Raw & processed datasets
├── 📂 models/                  # Trained model artifacts
├── 📂 notebooks/               # Experiment notebooks
├── 📂 docs/                    # Project documentation
├── 📂 reports/                 # Generated reports
├── 📂 references/              # Reference materials
│
├── 📂 src/
│   ├── 📂 connections/         # External service connections
│   ├── 📂 data/                # Data ingestion & processing
│   ├── 📂 features/            # Feature engineering
│   ├── 📂 logger/              # Logging utilities
│   ├── 📂 model/               # ML training & evaluation
│   └── 📂 visualization/       # Visual analytics
│
├── 📂 flask_app/               # Model serving application
│
├── 📂 tests/                   # Unit & integration tests
│
├── 📂 .github/workflows/       # CI/CD pipelines
│
├── 🐳 Dockerfile              # Containerization
├── ⚙️ dvc.yaml                # DVC pipeline
├── ⚙️ params.yaml             # Pipeline parameters
├── 📦 requirements.txt        # Dependencies
├── 🔧 setup.py                # Package configuration
├── 📖 README.md               # Project documentation
└── 📄 LICENSE
```

---

# 🔄 MLOps Workflow

## 1️⃣ Data Versioning

* Dataset tracked using DVC
* Version control for datasets
* Remote storage configured on AWS S3
* Reproducible pipeline execution

### DVC Pipeline

```bash
dvc repro
```

### Check Status

```bash
dvc status
```

---

## 2️⃣ Experiment Tracking

ML experiments are logged into:

* MLflow
* DagsHub

Tracked Parameters:

* Hyperparameters
* Metrics
* Artifacts
* Model Versions

---

## 3️⃣ Model Training Pipeline

### Pipeline Stages

```text
Data Ingestion
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Registration
```

---

## 4️⃣ AWS S3 Integration

Used for:

* Dataset Storage
* Model Artifacts
* DVC Remote Storage

```bash
dvc remote add -d myremote s3://bucket-name
```

---

# 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t mlops-app:latest .
```

### Run Container

```bash
docker run -p 8888:5000 mlops-app:latest
```

### Run with Environment Variables

```bash
docker run \
-p 8888:5000 \
-e DAGSHUB_TOKEN=your_token \
mlops-app:latest
```

---

# ⚙️ CI/CD Pipeline

GitHub Actions automates:

### CI

* Code Validation
* Dependency Installation
* Testing
* Docker Build

### CD

* Push Docker Image to AWS ECR
* Deploy to AWS EKS

Workflow:

```text
Developer Push
      ↓
GitHub Actions
      ↓
Run Tests
      ↓
Build Docker Image
      ↓
Push to ECR
      ↓
Deploy to EKS
```

---

# ☁️ AWS Infrastructure

### Services Used

| Service        | Purpose                     |
| -------------- | --------------------------- |
| IAM            | Authentication              |
| S3             | Artifact Storage            |
| ECR            | Docker Registry             |
| EKS            | Kubernetes Cluster          |
| EC2            | Monitoring Servers          |
| CloudFormation | Infrastructure Provisioning |

---

# ☸️ Kubernetes Deployment

Application deployed on AWS EKS.

### Verify Cluster

```bash
kubectl get nodes
```

### Verify Pods

```bash
kubectl get pods
```

### Verify Services

```bash
kubectl get svc
```

---

# 📊 Monitoring & Observability

## Prometheus

Collects:

* Application Metrics
* System Metrics
* Kubernetes Metrics

### Access

```text
http://<prometheus-ip>:9090
```

---

## Grafana

Provides real-time dashboards for:

* Request Monitoring
* Resource Usage
* Application Health
* Infrastructure Metrics

### Access

```text
http://<grafana-ip>:3000
```

---

# 🔐 Security Features

* IAM-Based Authentication
* GitHub Secrets Management
* Environment Variables for Credentials
* Secure AWS Resource Access
* Containerized Deployment

---

# 🚀 CI/CD + Deployment Flow

```text
GitHub
   ↓
GitHub Actions
   ↓
Docker Build
   ↓
AWS ECR
   ↓
AWS EKS
   ↓
Load Balancer
   ↓
Production API
   ↓
Prometheus
   ↓
Grafana Dashboard
```

---

# 📈 Business Value

This project demonstrates expertise in:

* Machine Learning Engineering
* MLOps
* Cloud Computing
* Kubernetes
* CI/CD Automation
* Monitoring & Observability
* Infrastructure Automation

It showcases how enterprise-grade ML systems can be built, deployed, monitored, and scaled in production environments.

---

# 👨‍💻 Author

**Suraj Kumar**

AI/ML Engineer | MLOps Engineer | Generative AI Developer

📧 Email: srsuraj009@gmail.com

🔗 LinkedIn: www.linkedin.com/in/aideveloperontop

Cont No. 9801263970


---

⭐ If you found this project useful, consider giving it a star.

```


```
