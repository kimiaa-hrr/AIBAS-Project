# FROM python:3.10-slim

# RUN pip install --no-cache-dir tensorflow==2.15.0 pandas numpy scikit-learn

# WORKDIR /app

# COPY apply_ann.py .

# ENTRYPOINT ["python", "apply_ann.py"]


FROM python:3.11-slim

RUN RUN pip install --no-cache-dir tensorflow==2.15.0 pandas numpy scikit-learn

RUN mkdir -p /codeBase

COPY apply_ann.py /codeBase/


