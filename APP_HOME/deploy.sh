#!/usr/bin/env bash

#PROJECT ID
PROJECT_ID="dami-dev-250916"
REGION="europe-west1"
CONTAINER_NAME="build_deploy_ml_model_tutorial"
CONTAINER_URL=gcr.io/${PROJECT_ID}/${CONTAINER_NAME}

#IAM
#SERVICE_ACCOUNT=""

gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Build the container
gcloud builds submit --tag ${CONTAINER_URL}

#Run the container on cloud run
gcloud beta run deploy --image ${CONTAINER_URL} --platform managed --allow-unauthenticated --service-account=${SERVICE_ACCOUNT}