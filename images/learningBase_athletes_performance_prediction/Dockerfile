FROM busybox

# Create folders in the image (outside /tmp)
RUN mkdir -p /learningBase/train /learningBase/validation

# Copy files into the image folders
COPY training_data.csv /learningBase/train/
COPY test_data.csv /learningBase/validation/
COPY README.md /learningBase/

