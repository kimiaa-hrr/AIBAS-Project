# Project Name

**Course:** M. Grum: Advanced AI-based Application Systems  
**Repository Ownership:**  
- Team Memberss: 

    -- Kimia Hariri, 
    -- Member 2 ---

---

## Description

Briefly describe your project here.  
Include the goal, approach, and any key algorithms or techniques used.

---

## Directory Structure

.
├── code/
│   ├── dataCleaning.py
│   ├── dataPreparation.py
│   └── dataScraping.py
├── data/dataset/
│   ├── activation_data.csv
│   ├── cleaned_dataset.csv
│   ├── joint_data_collection.csv
│   ├── README.md
│   ├── scrapedDataset.csv
│   ├── test_data.csv
│   └── training_data.csv
├── images/
│   ├── activationBase_athletes_performance_p.../
│   │   ├── activation_data.csv
│   │   ├── docker-compose.yml
│   │   ├── Dockerfile
│   │   └── README.md
│   └── learningBase_athletes_performance_pre.../
│       ├── docker-compose.yml
│       ├── Dockerfile
│       ├── README.md
│       ├── test_data.csv
│       └── training_data.csv
└── README.md




## How to get the image

Pull the image from Docker Hub:

```bash
docker pull kimiahr/learningbase_athletes_performance_prediction:latest

docker pull kimiahr/activationbase_athletes_performance_prediction:latest

```
## How to get the image for running the code
```bash
docker pull kimiahr/tensorflow_custom:latest
```

## How to Run the TensorFlow container interactively
```bash

docker run -it --rm \
  -v ai_system:/tmp \
  -v /path/to/your/project:/app \
  kimiahr/tensorflow_custom:latest \
  /bin/bash

```
