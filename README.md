## SETUP
pip install flask flask-sqlalchemy requests werkzeug
pip install flask flask-sqlalchemy pyjwt werkzeug



Quick Start with Electron:

Create the Electron wrapper
Package with electron-builder:
npm install electron-builder --save-dev
npm run build  # Creates distributable files




# EC2 Deployment Guide for Your Interview App

## Step 1: Launch EC2 Instance

### 1.1 Create EC2 Instance
1. Go to AWS Console → EC2 → Launch Instance
2. **Name**: `app-prototype`
3. **AMI**: Ubuntu Server 22.04 LTS (Free tier eligible)
4. **Instance type**: t2.micro (Free tier eligible)
5. **Key pair**: Create new or use existing (.pem file)
6. **Storage**: 8GB GP2 (default is fine)

### 1.2 Configure Security Group
Create/edit security group with these rules:
- **SSH**: Port 22, Source: My IP (your current IP)
- **HTTP**: Port 80, Source: 0.0.0.0/0 (anywhere)
- **HTTPS**: Port 443, Source: 0.0.0.0/0 (anywhere)
- **Custom**: Port 5000, Source: 0.0.0.0/0 (your app port)

## Step 2: Connect to EC2 Instance

```bash
# Make your key file secure
chmod 400 your-key-file.pem

# SSH into your instance
ssh -i your-key-file.pem ubuntu@YOUR-EC2-PUBLIC-IP
```

## Step 3: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python, Node.js, Git, and Nginx
sudo apt install python3 python3-pip python3-venv nodejs npm git nginx -y

# Install PM2 for process management (optional but recommended)
sudo npm install -g pm2
```

## Step 4: Deploy Your Application

### 4.1 Upload Your Code
**Option A: Using Git (Recommended)**
```bash
# If your code is on GitHub/GitLab
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

**Option B: Using SCP to upload files**
```bash
# From your local machine (separate terminal)
scp -i your-key-file.pem -r /path/to/your/local/app ubuntu@YOUR-EC2-PUBLIC-IP:~/
```

### 4.2 Set Up Python Environment
```bash
# Navigate to your app directory
cd ~/your-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# If you don't have requirements.txt, create it:
pip freeze > requirements.txt
```

### 4.3 Set Up Directories and Permissions
```bash
# Create uploads directory if it doesn't exist
mkdir -p uploads
mkdir -p instance  # for SQLite database

# Set proper permissions
chmod 755 uploads
chmod 755 instance
chmod 644 *.db  # if you have existing database file

# Make sure ubuntu user owns everything
sudo chown -R ubuntu:ubuntu ~/your-app
```

## Step 5: Test Your Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run your app (replace with your actual run command)
python3 app.py

# Test in another terminal or browser
curl http://localhost:5000
```

## Step 7: Configure Nginx (Optional - for port 80)

```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/interview-app
```

Add this content:
```nginx
server {
    listen 80;
    server_name YOUR-EC2-PUBLIC-IP;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:
```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/interview-app /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test and reload Nginx
sudo nginx -t
sudo systemctl reload nginx
```

## Step 8: Final Testing

### 8.1 Test Your App
```bash
# Direct app access
curl http://YOUR-EC2-PUBLIC-IP:5000

# Through Nginx (if configured)
curl http://YOUR-EC2-PUBLIC-IP
```

Your app will be accessible at:
- **Direct access**: `http://YOUR-EC2-PUBLIC-IP:5000`
- **Through Nginx**: `http://YOUR-EC2-PUBLIC-IP` (port 80)

## Troubleshooting

### Common Issues:
1. **App won't start**: Check Python dependencies and file permissions
2. **Can't access from browser**: Verify security group settings
3. **Database errors**: Check SQLite file permissions and directory ownership
4. **File upload fails**: Verify uploads directory exists and is writable