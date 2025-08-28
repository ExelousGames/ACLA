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
