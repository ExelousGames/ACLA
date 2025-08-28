FROM mongo:7.0.21-rc0-jammy

# Install cron and required tools
RUN apt-get update && \
    apt-get install -y cron && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy backup script, chmod +x add executable permissions to a file
COPY backup.sh /backup.sh
RUN chmod +x /backup.sh

# Copy restore script
COPY restore.sh /restore.sh
RUN chmod +x /restore.sh

# Copy cron job file, chmod 0644: you can read and write the file or directory and other users can only read it.
COPY mongo-cron /etc/cron.d/mongo-cron
RUN chmod 0644 /etc/cron.d/mongo-cron

# Create the log file
RUN touch /var/log/cron.log

#TODO add timed backup scrip