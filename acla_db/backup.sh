#!/bin/bash


DATE=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_DIR="/backups/acla_$DATE"

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

# Compress backup - tar [options] <archive_name> <files_or_directories> 
tar -zcvf "/backups/acla_$DATE.tar.gz" -C $BACKUP_DIR . 2>> /var/log/backup.log
rm -rf $BACKUP_DIR

# Delete backups older than 3 days
find /backups -name "*.tar.gz" -mtime +3 -delete 2>> /var/log/backup.log

# Keep only the 5 newest .tar.gz backups in /backups
# 1. Finds all matching backup files in /backups (not subdirectories), Outputs: [timestamp] [fullpath] for each file - find /backups -maxdepth 1 -name "acla_*.tar.gz" -type f -printf "%T@ %p\n"
# 2. Sorts by first column (timestamp) in reverse numerical order (newest first) - sort -k1 -rn
# 3. Keeps first 5 lines (NR≤5) - awk 'NR>5 {system("rm -f " $2)}'
find /backups -maxdepth 1 -name "acla_*.tar.gz" -type f -printf "%T@ %p\n" | sort -k1 -rn | awk 'NR>5 {system("rm -f " $2)}'

echo "$(date) - Backup completed" >> /var/log/backup.log