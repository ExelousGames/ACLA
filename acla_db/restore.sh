#!/bin/bash
#
# restore.sh - Restore the newest ACLA MongoDB backup from /backups.
#
# HOW TO USE
#   This script is baked into the mongo image (see Dockerfile) and is intended
#   to be run from inside the running container. It auto-selects the newest
#   acla_*.tar.gz file in /backups and restores it, dropping existing
#   collections in the target database first (--drop).
#
#   Run from the host:
#     docker exec -it mongodb_c /restore.sh
#
#   Or from inside the running container:
#     /restore.sh
#
#   To restore a specific (non-newest) backup, either temporarily move newer
#   files out of /backups, or run mongorestore by hand against the extracted
#   dump.
#
# REQUIRED ENV VARS (provided by docker-compose / the mongo image)
#   MONGO_INITDB_ROOT_USERNAME  - root user used to authenticate
#   MONGO_INITDB_ROOT_PASSWORD  - root password
#   MONGO_DATEBASE              - name of the database to restore into (note spelling)
#
# WARNING
#   --drop wipes existing collections in $MONGO_DATEBASE before restoring.
#   Take a fresh backup (./backup.sh) first if the current data matters.
#
# INPUT
#   backups/acla_<YYYY-MM-DD_HH-MM-SS>.tar.gz   newest matching file is used
#
# RELATED
#   backup.sh in this directory creates the archives this script consumes.

# Configuration
CONTAINER_NAME="mongodb_c"
BACKUP_GLOB="acla_*.tar.gz"  # Backup file pattern

# Find newest backup inside container
NEWEST_BACKUP=$(find backups -name "$BACKUP_GLOB" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$NEWEST_BACKUP" ]; then
  echo "Error: No backup files found in /backups"
  exit 1
fi

echo "Found newest backup: $(basename "$NEWEST_BACKUP")"

# Extract just the timestamp portion (for the extracted folder)
BACKUP_FOLDER=$(basename "$NEWEST_BACKUP" .tar.gz)

# Create temp directory inside container
TEMPDIR="/tmp/mongorestore_$(date +%s)"
mkdir -p "$TEMPDIR"

# Extract backup
echo "Extracting backup..."
tar -xzvf "$NEWEST_BACKUP" -C "$TEMPDIR" || {
  echo "Error extracting backup"
  exit 1
}

# Restore to MongoDB
echo "Restoring to database '$MONGO_DATEBASE'..."
mongorestore \
  --username "$MONGO_INITDB_ROOT_USERNAME" \
  --password "$MONGO_INITDB_ROOT_PASSWORD" \
  --authenticationDatabase admin \
  --db "$MONGO_DATEBASE" \
  --drop \
  "$TEMPDIR/$MONGO_DATEBASE" || {
    echo "Error during mongorestore"
    exit 1
}

# Cleanup
echo "Cleaning up..."
rm -rf "$TEMPDIR"

echo "Successfully restored $(basename "$NEWEST_BACKUP") to $MONGO_DATEBASE"