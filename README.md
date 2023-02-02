# Machine Learning Model Deployment Guide

This repository contains my guide to deploy ML models. The repository was forked from 

[deploying-machine-learning-models](https://github.com/trainindata/deploying-machine-learning-models)

which contains the companion code of the Udemy course [Deployment of Machine Learning Models](https://www.udemy.com/course/deployment-of-machine-learning-models) by Soledad Galli & Christopher Samiullah.

The guide or notes for my future self done after following the course are in: `./ML_Deployment_Guide.md`.

## Notes on the Contents of the Repository

- Presentations: provided as a Dropbox download link, located in `./udemy_ml_deployment/deployment_of_ML_presentations`; but not committed.
- Datasets: downloaded from [kaggle](www.kaggle.com): [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data); located in `./data/house-prices-advanced-regression-techniques`; but not committed.

## Overview of the Contents of the Course

1. Overview of Model Deployment
2. Machine Learning System Architecture
3. Research Environment: Developing a Machine Learning Model
4. Packaging the Model for Production
5. Serving and Deploying the Model via REST API - FastAPI
6. Continuous Integration and Deployment Pipelines - CicleCI
7. Deploying the ML API with Containers
8. Deploying to IaaS (AWS ECS)

## How to Start?

Continue in [`./ML_Deployment_Guide.md`](ML_Deployment_Guide.md) for the detailed guide/notes.

Also check these links:

- Through deployment example: [census_model_deployment_fastapi](https://github.com/mxagar/census_model_deployment_fastapi).
- My personal notes on the [Udacity MLOps](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) nanodegree: [mlops_udacity](https://github.com/mxagar/mlops_udacity). The module guide about deployment: [`MLOpsND_Deployment.md`](https://github.com/mxagar/mlops_udacity/blob/main/03_Deployment/MLOpsND_Deployment.md)

## Relevant Links

- Check this example deployment: [census_model_deployment_fastapi](https://github.com/mxagar/census_model_deployment_fastapi)
- Check my personal notes on the [Udacity MLOps](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) nanodegree: [mlops_udacity](https://github.com/mxagar/mlops_udacity):
  - The module guide about deployment: [`MLOpsND_Deployment.md`](https://github.com/mxagar/mlops_udacity/blob/main/03_Deployment/MLOpsND_Deployment.md)
  - Example and exercise repository related to the topic of deployment: [mlops-udacity-deployment-demos](https://github.com/mxagar/mlops-udacity-deployment-demos).
- My guide on CI/DC: [cicd_guide](https://github.com/mxagar/cicd_guide)
- My boilerplate for reproducible ML pipelines using [MLflow](https://www.mlflow.org/) and [Weights & Biases](https://wandb.ai/site): [music_genre_classification](https://github.com/mxagar/music_genre_classification).
- A very [simple Heroku deployment](https://github.com/mxagar/data_science_python_tools/tree/main/19_NeuralNetworks_Keras/19_11_Keras_Deployment) with the Iris dataset and using Flask as API engine.
- Notes on how to transform research code into production-level packages: [customer_churn_production](https://github.com/mxagar/customer_churn_production).
- My summary of data processing and modeling techniques: [eda_fe_summary](https://github.com/mxagar/eda_fe_summary).

## Authorship

Notes by Mikel Sagardia, 2022.  
No guarantees.