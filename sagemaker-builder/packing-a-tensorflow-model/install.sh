#!/usr/bin/env bash

python -m venv new-env
source new-env/bin/activate

pip install pip -U
pip install -r requirements.txt

# build the image
python run_benchmarks.py   --instance_count 4 --role SageMakerRole --tag tensorflow-hvd:latest --base-image mvsusp/hvd-benchmark \
  --aws_account your-account --region us-west-2 --subnet your-subnet --security_group your-sg

# dont build
python run_benchmarks.py   --instance_count 4 --role SageMakerRole --tag tensorflow-hvd:latest --base-image mvsusp/hvd-benchmark --aws-account your-account --region us-west-2 --subnet your-subnet --security_group syour-sg --no-build