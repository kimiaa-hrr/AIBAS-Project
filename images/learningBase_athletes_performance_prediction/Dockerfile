FROM busybox

# Create folders
RUN mkdir -p /tmp/learningBase/train /tmp/learningBase/validation

# Copy files
COPY training_data.csv /tmp/learningBase/train/
COPY test_data.csv /tmp/learningBase/validation/
COPY README.md /tmp/learningBase/

CMD ["sh"]
