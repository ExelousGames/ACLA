#!/bin/bash


DATE=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_DIR="/backups/$DATE"

# logging
echo "$(date) - Starting backup" >> /var/log/backup.log

# Create backup, 
# --out $BACKUP_DIR :The output directory where MongoDB will write the backup files
# 2>: Redirects file descriptor 2 (stderr - standard error output)
# >>: Appends to the file rather than overwriting
mongodump \
  --username $MONGO_INITDB_ROOT_USERNAME \
  --password $MONGO_INITDB_ROOT_PASSWORD \
  --authenticationDatabase admin \
  --db $MONGO_DATEBASE \
  --out $BACKUP_DIR 2>> /var/log/backup.log

# Compress backup
tar -zcvf "/backups/$DATE.tar.gz" $BACKUP_DIR 2>> /var/log/backup.log
rm -rf $BACKUP_DIR

# Delete backups older than 3 days
find /backups -name "*.tar.gz" -mtime +3 -delete 2>> /var/log/backup.log

echo "$(date) - Backup completed" >> /var/log/backup.log