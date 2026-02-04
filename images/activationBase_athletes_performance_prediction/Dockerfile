FROM busybox

# Create folder in the image (outside /tmp)
RUN mkdir -p /activationBase

# Copy files into the image folder
COPY activation_data.csv /activationBase/
COPY README.md /activationBase/
