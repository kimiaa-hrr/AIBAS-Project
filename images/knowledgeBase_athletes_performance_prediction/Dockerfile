FROM busybox

# Create folder in the image (outside /tmp)
RUN mkdir -p /knowledgeBase

# Copy files into the image folder
COPY currentOlsSolution.pkl /knowledgeBase/
COPY ann_model.keras /knowledgeBase/

COPY README.md /knowledgeBase/
