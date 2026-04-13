#!/usr/bin/env bash
# Enable APIs needed for AnkReg (run once per project).
# Prerequisites: gcloud auth login && gcloud config set project YOUR_PROJECT

set -euo pipefail
PROJECT="${GCP_PROJECT:-}"
if [[ -z "$PROJECT" ]]; then
  echo "Set GCP_PROJECT in the environment." >&2
  exit 1
fi

gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  --project="$PROJECT"

echo "APIs enabled for project $PROJECT"
