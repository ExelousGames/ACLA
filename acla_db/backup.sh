#!/bin/bash
#
# backup.sh - Dump and compress the ACLA MongoDB database.
#
# HOW TO USE
#   This script is baked into the mongo image (see Dockerfile) and run by cron
#   daily at 02:00 (see mongo-cron). You normally don't invoke it by hand.
#
#   Run a one-off backup manually (from the host):
#     docker exec -it mongodb_c /backup.sh
#
#   Or from inside the running container:
#     /backup.sh
#
# REQUIRED ENV VARS (provided by docker-compose / the mongo image)
#   MONGO_INITDB_ROOT_USERNAME  - root user used to authenticate
#   MONGO_INITDB_ROOT_PASSWORD  - root password
#   MONGO_DATEBASE              - name of the database to dump (note spelling)
#
# OUTPUT
#   /backups/acla_<YYYY-MM-DD_HH-MM-SS>.tar.gz   compressed dump
#   /var/log/backup.log                          run log (start/finish + errors)
#
# RETENTION
#   Only the 5 newest acla_*.tar.gz files in /backups are kept; older ones are
#   deleted at the end of each run.
#
# RESTORE
#   Use restore.sh in this directory; it auto-picks the newest backup.

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
#find /backups -name "*.tar.gz" -mtime +3 -delete 2>> /var/log/backup.log

# Keep only the 5 newest .tar.gz backups in /backups
# 1. Finds all matching backup files in /backups (not subdirectories), Outputs: [timestamp] [fullpath] for each file - find /backups -maxdepth 1 -name "acla_*.tar.gz" -type f -printf "%T@ %p\n"
# 2. Sorts by first column (timestamp) in reverse numerical order (newest first) - sort -k1 -rn
# 3. Keeps first 5 lines (NR≤5) - awk 'NR>5 {system("rm -f " $2)}'
find /backups -maxdepth 1 -name "acla_*.tar.gz" -type f -printf "%T@ %p\n" | sort -k1 -rn | awk 'NR>5 {system("rm -f " $2)}'

echo "$(date) - Backup completed" >> /var/log/backup.log