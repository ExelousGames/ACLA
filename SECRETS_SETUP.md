# Environment Secrets Setup

This document explains how to securely manage sensitive environment variables like API keys.

## Quick Setup

1. **Copy the secrets template:**
   ```bash
   cp .env.secrets.template .env.secrets
   ```

2. **Edit the secrets file:**
   ```bash
   # Open .env.secrets and replace the placeholder values with your actual keys
   nano .env.secrets  # or use your preferred editor
   ```

3. **Verify the file is ignored by git:**
   ```bash
   git status  # .env.secrets should not appear in the list
   ```

## Google Cloud Platform (GCP) Setup

For Gemini Batch API functionality, you need to configure GCP credentials:

### 1. Service Account Setup

1. **Place your GCP service account JSON file at:**
   ```
   acla_ai_service/service_src/gcp-service-account.json
   ```

2. **Required IAM permissions for the service account:**
   - `storage.objects.create` (to upload batch input files)
   - `storage.objects.get` (to read batch results)
   - `storage.buckets.get` (to access bucket metadata)

### 2. Environment Variables

Add these to your `.env.secrets` file:

```bash
# GCS bucket name (without gs:// prefix)
GEMINI_BATCH_GCS_BUCKET=your-bucket-name

# Path to GCP service account file (inside container)
GOOGLE_APPLICATION_CREDENTIALS=/app/service_src/gcp-service-account.json

# Gemini API key (for non-batch API calls)
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Docker Integration

The `docker-compose.dev.yaml` automatically:
- Mounts the `acla_ai_service` directory to `/app` (includes `service_src/`)
- Reads `GOOGLE_APPLICATION_CREDENTIALS` from `.env.secrets`
- Passes `GEMINI_BATCH_GCS_BUCKET` to the container

### 4. Troubleshooting GCS Authentication

If you see "File gcp-service-account.json was not found":
1. Verify the file exists at `acla_ai_service/service_src/gcp-service-account.json`
2. Restart Docker containers: `docker-compose -f docker-compose.dev.yaml up -d ai_service`

If you see "Permission denied" errors:
1. Verify the service account has the required IAM roles
2. Check the bucket name is correct in `.env.secrets`

## Security Best Practices

### ✅ DO:
- Keep sensitive data in `.env.secrets` file
- Add all secret files to `.gitignore`
- Use strong, unique API keys
- Rotate API keys regularly
- Share secrets through secure channels (not email/chat)

### ❌ DON'T:
- Commit `.env.secrets` to version control
- Share API keys in public channels
- Use the same API key across multiple environments
- Store secrets in regular environment files

## File Structure

```
.dev.env          # Non-sensitive development config (committed)
.env.secrets      # Sensitive data like API keys (NOT committed)
.env.secrets.template  # Template for team setup (committed)
```

## Production Deployment

For production environments, consider using:
- **Docker Secrets** (Docker Swarm)
- **Kubernetes Secrets** (Kubernetes)
- **AWS Secrets Manager** (AWS)
- **Azure Key Vault** (Azure)
- **HashiCorp Vault** (Multi-cloud)

## Troubleshooting

### "OPENAI_API_KEY not found" error:
1. Ensure `.env.secrets` file exists
2. Check that it contains `OPENAI_API_KEY=your_actual_key`
3. Restart Docker containers: `docker-compose -f docker-compose.dev.yaml up -d`

### API key not working:
1. Verify the key is valid on OpenAI's platform
2. Check for extra spaces or characters in the key
3. Ensure the key has appropriate permissions
