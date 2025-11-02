# AWS Deployment Guide

This directory contains configurations and scripts for deploying SmartPantry to AWS.

## Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  CloudFront     │  (Optional CDN)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ALB/ELB       │  (Load Balancer)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EC2 Instance   │  (Application Server)
│  - Docker       │
│  - Gradio App   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌─────┐
│  S3   │ │ RDS │  (Optional)
└───────┘ └─────┘
```

## Prerequisites

1. AWS Account with appropriate permissions
2. AWS CLI installed and configured
3. Docker installed locally
4. (Optional) Terraform for infrastructure as code

## Deployment Options

### Option 1: EC2 with Docker

#### Step 1: Launch EC2 Instance

```bash
# Launch GPU instance for better performance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=smartpantry-app}]'
```

#### Step 2: Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Step 3: Deploy Application

```bash
# Clone repository
git clone https://github.com/your-username/smartpantry.git
cd smartpantry

# Create .env file
cp .env.example .env
# Edit .env with production values

# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

#### Step 4: Configure Security Group

```bash
# Allow HTTP/HTTPS and application port
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 7860 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0
```

### Option 2: ECS with Fargate

#### Step 1: Create ECR Repository

```bash
# Create repository
aws ecr create-repository --repository-name smartpantry

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push image
docker build -t smartpantry .
docker tag smartpantry:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/smartpantry:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/smartpantry:latest
```

#### Step 2: Create ECS Task Definition

See `ecs-task-definition.json` in this directory.

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://deployment/aws/ecs-task-definition.json
```

#### Step 3: Create ECS Service

```bash
# Create service
aws ecs create-service \
  --cluster smartpantry-cluster \
  --service-name smartpantry-service \
  --task-definition smartpantry-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}"
```

### Option 3: Lambda + API Gateway (Serverless)

For serverless deployment with AWS Lambda:

1. Package application with dependencies
2. Create Lambda function with container image
3. Set up API Gateway
4. Configure S3 for model storage

See `lambda/` subdirectory for Lambda-specific code.

## S3 Configuration

Store models and data in S3:

```bash
# Create S3 bucket
aws s3 mb s3://smartpantry-data

# Upload models
aws s3 cp models/ s3://smartpantry-data/models/ --recursive

# Upload recipe database
aws s3 cp data/recipes/ s3://smartpantry-data/recipes/ --recursive

# Set bucket policy for EC2 access
aws s3api put-bucket-policy --bucket smartpantry-data --policy file://s3-bucket-policy.json
```

## Auto Scaling

Configure auto-scaling for ECS or EC2:

```bash
# Create auto-scaling group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name smartpantry-asg \
  --launch-template LaunchTemplateId=lt-xxxxx \
  --min-size 1 \
  --max-size 5 \
  --desired-capacity 2 \
  --vpc-zone-identifier "subnet-xxxxx,subnet-yyyyy"

# Create scaling policies
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name smartpantry-asg \
  --policy-name scale-up \
  --scaling-adjustment 1 \
  --adjustment-type ChangeInCapacity
```

## Monitoring

### CloudWatch Logs

```bash
# Create log group
aws logs create-log-group --log-group-name /aws/smartpantry

# View logs
aws logs tail /aws/smartpantry --follow
```

### CloudWatch Metrics

Monitor:
- CPU utilization
- Memory usage
- Request count
- Response time
- Error rate

## Cost Optimization

1. **Use Spot Instances**: Save up to 90% on EC2 costs
2. **Auto-scaling**: Scale down during low traffic
3. **S3 Lifecycle Policies**: Move old data to cheaper storage tiers
4. **CloudFront CDN**: Reduce data transfer costs
5. **Reserved Instances**: Commit to 1-3 years for discounts

## Estimated Costs (Monthly)

**Small Deployment (Development)**
- EC2 t3.medium: ~$30
- S3 (100GB): ~$2.3
- Data Transfer: ~$10
- **Total: ~$42/month**

**Medium Deployment (Production)**
- EC2 g4dn.xlarge (with spot): ~$75
- ALB: ~$22
- S3 (500GB): ~$11.5
- CloudWatch: ~$10
- Data Transfer: ~$50
- **Total: ~$168/month**

**Large Deployment (High Traffic)**
- ECS Fargate (2 tasks): ~$150
- ALB: ~$22
- S3 (1TB): ~$23
- RDS: ~$100
- CloudFront: ~$50
- CloudWatch: ~$25
- **Total: ~$370/month**

## Troubleshooting

### Application not accessible
- Check security group rules
- Verify instance is running
- Check application logs

### Out of memory
- Increase instance type
- Optimize model sizes
- Enable swap space

### Slow inference
- Use GPU instances (g4dn family)
- Enable model caching
- Use Application Load Balancer for multiple instances

## Security Best Practices

1. Use IAM roles instead of access keys
2. Enable encryption at rest (S3, EBS)
3. Use VPC and private subnets
4. Enable CloudTrail for auditing
5. Regular security patches and updates
6. Use Secrets Manager for sensitive data
7. Implement WAF rules for API protection

