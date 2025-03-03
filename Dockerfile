# Use the ARM-based ultralytics image as the base
FROM ultralytics/ultralytics:latest-jetson-jetpack4

# Install necessary packages: git for cloning and tmux for running the process
RUN apt-get update && apt-get install -y git tmux

# Clone the ml_jetson repository into /ml_jetson
RUN git clone https://github.com/jalan-subham/ml_jetson.git /ml_jetson

# Set the working directory
WORKDIR /ml_jetson

# Install Python dependencies
# Assumes your repository has a requirements.txt file; also install uvicorn
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install uvicorn

# Expose port 8000 (adjust if uvicorn uses a different port)
EXPOSE 8000

# Create an entrypoint script that starts uvicorn in a detached tmux session
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'tmux new-session -d -s uvicorn_session "uvicorn main:app --host 0.0.0.0"' >> /entrypoint.sh && \
    echo 'tail -f /dev/null' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Use the entrypoint script as the container's entrypoint
ENTRYPOINT ["/entrypoint.sh"]

