#!/usr/bin/env bash
# Example: bind a Cloud Run service account to invoke Vertex AI and BigQuery.
# Replace SERVICE_ACCOUNT, SERVICE_NAME, and REGION.

set -euo pipefail
PROJECT="${GCP_PROJECT:?}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:?}"  # e.g. ankrag-runner@PROJECT.iam.gserviceaccount.com

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin"

echo "Granted Vertex, BigQuery, and Storage roles to ${SERVICE_ACCOUNT}"
