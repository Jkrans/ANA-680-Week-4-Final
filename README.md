# ANA 680 Final Project – Iris Classification Pipeline

## Problem Statement
The goal of this project is to classify iris flowers into one of three species  
(Setosa, Versicolor, Virginica) based on four numerical features:

- Sepal length  
- Sepal width  
- Petal length  
- Petal width  

Machine learning is appropriate because the relationship between measurements and species is not rule-based and must be learned from data.

---

## Dataset

- **Dataset:** Iris Dataset  
- **Source:** UCI Machine Learning Repository  
- **Loaded via:** `sklearn.datasets.load_iris()`  

The dataset contains 150 samples evenly distributed across three species.

---

## Model Development

Performed in `iris_project.ipynb`:

- Data exploration (EDA)
- Train / Validation / Test split (70 / 15 / 15)
- Logistic Regression model
- Evaluation metrics:
  - Accuracy
  - Confusion Matrix
  - Classification Report

Final test accuracy: ~96%

The trained model is saved as:

---

## Deployment Stack

This project demonstrates a full ML deployment pipeline:

1. **Local Deployment**
   - Flask app (`app.py`)
   - HTML frontend (`templates/index.html`)

2. **Containerization**
   - Dockerfile
   - Image pushed to Docker Hub
    https://hub.docker.com/r/jkrans/iris-final

3. **CI/CD**
   - GitHub Actions workflow
   - Automatic deployment to Heroku

4. **Cloud Deployment**
   - Heroku web app

5. **Kubernetes Deployment**
   - Deployment YAML
   - Service YAML

---

## Live App

https://jk-iris-final-e515cc35fd58.herokuapp.com/

---
## AWS SageMaker Studio Lab (Pending)

Studio Lab access has not been approved (requested >1 week ago).  
I requested on Saturday the 21st, they replied Monday morning saying
they got the request, but it has still not been approved and it's time
to turn this assignment in. 

## Technologies Used

- Python
- Scikit-learn
- Flask
- Docker
- GitHub Actions
- Heroku
- Kubernetes