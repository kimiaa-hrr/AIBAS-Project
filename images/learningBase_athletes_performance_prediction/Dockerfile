# Dockerfile
FROM busybox

# create directories
RUN mkdir -p /learningBase/train /learningBase/validation

# copy data
COPY training_data.csv /learningBase/train/training_data.csv
COPY test_data.csv /learningBase/validation/test_data.csv
COPY README.md /learningBase/README.md
