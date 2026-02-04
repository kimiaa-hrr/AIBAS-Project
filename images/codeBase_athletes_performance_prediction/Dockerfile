FROM python:3.11-slim

# Install required packages
RUN pip install --no-cache-dir \
    tensorflow==2.20.0 \
    pandas \
    numpy \
    scikit-learn \
    statsmodels  \
    torch


# Create folders for code and results
RUN mkdir -p /codeBase /tmp/results

# Copy both scripts into the image
COPY apply_ols.py /codeBase/
COPY apply_ann.py /codeBase/
