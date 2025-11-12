# Docker Deployment - Quick Guide

## 🐳 Build and Run

### Quick Commands

```bash
# Build the image
docker build -t ai-trading-bot:latest .

# Run the container
docker run -d \
  --name ai-trading-bot \
  --restart unless-stopped \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -p 8501:8501 \
  ai-trading-bot:latest

# View logs
docker logs -f ai-trading-bot
```

## 📊 Access

- **Dashboard**: http://localhost:8501
- **Logs**: `docker logs -f ai-trading-bot`

## 🔧 Container Details

### Entrypoint Options

The Dockerfile includes two entrypoint options:

#### 1. Python-based (Default) ✅

```dockerfile
ENTRYPOINT ["python3", "/app/docker-entrypoint.py"]
```

- **Pros**: Better process management, graceful shutdown, process monitoring
- **Recommended for**: Production deployments

#### 2. Bash-based (Alternative)

```dockerfile
ENTRYPOINT ["/bin/bash", "/app/start.sh"]
```

- **Pros**: Simpler, traditional approach
- **Use if**: You prefer shell scripts

To switch, edit the `Dockerfile` and uncomment the alternative entrypoint.

### What Runs Inside

1. **Trading Bot** (`main.py`) - Background process
2. **Dashboard** (`streamlit`) - Exposed on port 8501

### Environment Variables

Required in `.env` file:

```bash
ASSET_MODE=idss
MONGO_DB_PATH=...
MONGO_DB_USERNAME=...
MONGO_DB_PASSWORD=...
MONGO_DB_NAME=...
OPENROUTER_API_KEY=...
```

## 🛠️ Management Commands

```bash
# Stop container
docker stop ai-trading-bot

# Start container
docker start ai-trading-bot

# Restart container
docker restart ai-trading-bot

# Remove container
docker rm -f ai-trading-bot

# View logs (last 100 lines)
docker logs --tail 100 ai-trading-bot

# Execute command in container
docker exec -it ai-trading-bot bash

# Check container status
docker ps | grep ai-trading-bot
```

## 🔄 Update Deployment

```bash
# One-liner: rebuild and redeploy
docker build -t ai-trading-bot:latest . && \
  docker rm -f ai-trading-bot && \
  docker run -d --name ai-trading-bot --restart unless-stopped \
    --env-file .env -v $(pwd)/data:/app/data -p 8501:8501 \
    ai-trading-bot:latest
```

## 🐛 Troubleshooting

### Container exits immediately

```bash
# Check logs for errors
docker logs ai-trading-bot

# Run in foreground to see output
docker run --rm -it --env-file .env ai-trading-bot:latest
```

### Port already in use

```bash
# Find what's using port 8501
lsof -i :8501

# Kill the process or use different port
docker run -d ... -p 8502:8501 ai-trading-bot:latest
```

### Dashboard not accessible

```bash
# Check if container is running
docker ps

# Test from inside container
docker exec ai-trading-bot curl http://localhost:8501

# Check firewall/network settings
```

## 📋 Jenkins Integration

```groovy
pipeline {
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t ai-trading-bot:latest .'
            }
        }
        stage('Deploy') {
            steps {
                sh 'docker rm -f ai-trading-bot || true'
                sh '''
                    docker run -d --name ai-trading-bot \
                      --restart unless-stopped \
                      --env-file .env \
                      -v $(pwd)/data:/app/data \
                      -p 8501:8501 \
                      ai-trading-bot:latest
                '''
            }
        }
        stage('Verify') {
            steps {
                sh 'sleep 5'
                sh 'docker logs ai-trading-bot'
                sh 'docker ps | grep ai-trading-bot'
            }
        }
    }
}
```

## 💡 Tips

- Data persists in `./data` directory (mounted as volume)
- Container auto-restarts unless manually stopped
- Both bot and dashboard logs go to `docker logs`
- Bot continues running even when market is closed
- Use `--env-file` for secrets (don't bake into image)
