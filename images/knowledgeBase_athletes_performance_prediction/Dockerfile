FROM busybox

# Create folder
RUN mkdir -p /tmp/knowledgeBase

# Copy files
COPY currentOlsSolution.pkl /tmp/knowledgeBase/currentOlsSolution.pkl
# COPY currentSolution_ANN.h5 /tmp/knowledgeBase/currentSolution_ANN.h5
COPY README.md /tmp/knowledgeBase/Readme.md

CMD ["sh"]
