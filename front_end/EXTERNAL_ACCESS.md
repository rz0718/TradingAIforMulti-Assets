# Making Dashboard Accessible Externally

## Complete Guide to External Access Options

---

## Option 1: Streamlit Cloud (Recommended - FREE & Easy)

**Best for**: Permanent hosting, sharing with team/public

### Steps:
1. Push your code to GitHub (make sure `front_end/` folder is included)
2. Go to https://share.streamlit.io/
3. Sign in with GitHub
4. Click "New app"
5. Select your repository, branch, and `front_end/dashboard.py`
6. Click "Deploy"

**Advantages:**
- ✅ Free hosting
- ✅ Automatic HTTPS
- ✅ Auto-updates when you push to GitHub
- ✅ Custom subdomain (e.g., yourapp.streamlit.app)
- ✅ 24/7 availability

**Note**: You'll need to ensure your data files are accessible

---

## Option 2: Network Exposure (Quick Local Solution)

**Best for**: Temporary sharing on same network

### Using the provided script:
```bash
./run_dashboard_network.sh
```

This starts Streamlit with:
```bash
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Then share your computer's IP address:
- Find your local IP: `ipconfig getifaddr en0` (Mac) or `ipconfig` (Windows)
- Share URL: `http://YOUR_IP_ADDRESS:8501`
- Example: `http://192.168.1.100:8501`

**Advantages:**
- ✅ Quick setup (30 seconds)
- ✅ Works on local network (WiFi)
- ✅ No external dependencies

**Disadvantages:**
- ❌ Only works on same network
- ❌ Requires your computer to stay on
- ❌ Not accessible from internet

---

## Option 3: ngrok Tunnel (Internet Access)

**Best for**: Temporary demo/testing with anyone on internet

### Installation:
```bash
# Install ngrok
brew install ngrok  # Mac
# or download from https://ngrok.com/download

# Sign up at ngrok.com and get your auth token
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### Usage:
```bash
./run_dashboard_ngrok.sh
```

Or manually:
```bash
# Terminal 1: Start dashboard
streamlit run dashboard.py

# Terminal 2: Create tunnel
ngrok http 8501
```

You'll get a public URL like: `https://abc123.ngrok.io`

**Advantages:**
- ✅ Instant internet access
- ✅ HTTPS included
- ✅ No server setup needed
- ✅ Works behind firewalls

**Disadvantages:**
- ❌ URL changes each time (free plan)
- ❌ Requires your computer to stay on
- ❌ Free tier has limits (40 connections/minute)

---

## Option 4: Docker + Cloud Server

**Best for**: Production deployment with full control

### Quick Docker Setup:

Create `Dockerfile` in front_end:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY dashboard.py .
COPY .streamlit/ .streamlit/

# Copy data directory
COPY ../data/ /app/data/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.address", "0.0.0.0"]
```

Build and run:
```bash
docker build -t trading-dashboard .
docker run -p 8501:8501 -v $(pwd)/../data:/app/data trading-dashboard
```

### Deploy to Cloud:
- **Railway.app**: `railway up` (easiest)
- **Render.com**: Connect GitHub repo
- **DigitalOcean App Platform**: Click deploy
- **AWS ECS/Fargate**: Enterprise option
- **Google Cloud Run**: Serverless option
- **Azure Container Instances**: Microsoft option

---

## Option 5: VPS/Server Deployment

**Best for**: Complete control, professional deployment

### Basic Setup:
```bash
# On your server
git clone YOUR_REPO
cd TradingAIforMulti-Assets/front_end

# Install dependencies
pip install -r requirements.txt

# Run with external access
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501

# Use screen/tmux to keep it running
screen -S dashboard
streamlit run dashboard.py --server.address 0.0.0.0
# Press Ctrl+A, then D to detach
```

### With nginx reverse proxy (recommended):
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Add HTTPS with Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## 🔒 Security Considerations

### 1. Add Authentication

#### Basic Password Protection:
```python
# Add to dashboard.py
import streamlit_authenticator as stauth

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "your_secure_password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    
    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# Add to main():
if not check_password():
    st.stop()
```

### 2. Limit Access by IP (nginx)

```nginx
location / {
    # Only allow specific IPs
    allow 203.0.113.0/24;  # Your office network
    allow 198.51.100.5;     # Specific user
    deny all;
    
    proxy_pass http://localhost:8501;
}
```

### 3. Use Environment Variables

```python
import os

# Don't hardcode sensitive data
API_KEY = os.getenv("TRADING_API_KEY")
PASSWORD = os.getenv("DASHBOARD_PASSWORD")
```

---

## 📊 Data Considerations

### For Cloud Deployment:

**Option 1: Git-tracked data** (if data updates rarely)
```bash
git add data/*.csv
git commit -m "Add data files"
```

**Option 2: Cloud storage** (recommended for production)
```python
# Update dashboard.py to read from S3/GCS
import boto3

def load_data():
    s3 = boto3.client('s3')
    s3.download_file('your-bucket', 'data/ai_decisions.csv', '/tmp/ai_decisions.csv')
    df = pd.read_csv('/tmp/ai_decisions.csv')
    return df
```

**Option 3: API endpoint** (best for real-time)
```python
# Your trading bot exposes API
# Dashboard fetches data via HTTP
import requests
data = requests.get('https://your-bot-api.com/portfolio').json()
df = pd.DataFrame(data)
```

**Option 4: Database** (enterprise solution)
```python
import psycopg2
import pandas as pd

def load_data():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    df = pd.read_sql("SELECT * FROM ai_decisions", conn)
    return df
```

---

## 📋 Comparison Table

| Method | Setup | Cost | Uptime | Speed | Best For |
|--------|-------|------|--------|-------|----------|
| **Streamlit Cloud** | Easy | Free | 24/7 | Medium | Teams, demos |
| **Network Mode** | Instant | Free | When on | Fast | Local sharing |
| **ngrok** | Easy | Free | When on | Medium | Quick demos |
| **Docker + Cloud** | Medium | $5-50/mo | 24/7 | Fast | Production |
| **VPS** | Hard | $5-20/mo | 24/7 | Fast | Full control |

---

## 🚀 Recommendations

### For Different Use Cases:

**Quick Demo (< 1 hour)**
→ Use Network Mode or ngrok

**Team Dashboard (ongoing)**
→ Use Streamlit Cloud (free) or Docker + Render.com

**Production (enterprise)**
→ Use VPS with nginx, HTTPS, and authentication

**Development/Testing**
→ Use Local Mode

---

## 🆘 Troubleshooting

### Port Already in Use
```bash
# Find what's using port 8501
lsof -i :8501

# Kill the process
kill -9 PID
```

### Firewall Blocking Access
```bash
# Mac
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/streamlit

# Linux (ufw)
sudo ufw allow 8501

# Linux (firewalld)
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

### Cannot Access from External Network
1. Check `--server.address` is set to `0.0.0.0`
2. Check router firewall settings
3. Check if ISP blocks incoming connections
4. Try using ngrok instead

### Slow Performance
1. Reduce auto-refresh interval
2. Implement data caching
3. Use lighter components
4. Enable compression in config.toml

---

## 📚 Additional Resources

- [Streamlit Deployment Documentation](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
- [ngrok Documentation](https://ngrok.com/docs)
- [Docker Documentation](https://docs.docker.com/)
- [nginx Documentation](https://nginx.org/en/docs/)

---

**Need more help?** Check `SHARING_GUIDE.md` for quick reference or `INDEX.md` for all documentation.
