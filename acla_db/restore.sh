#!/bin/bash
# restore.sh - Auto-restores newest ACLA database backup from /backups

# Configuration
CONTAINER_NAME="mongodb"
BACKUP_GLOB="acla_*.tar.gz"  # Backup file pattern

# Find newest backup inside container
NEWEST_BACKUP=$(docker exec $CONTAINER_NAME \
  find /backups -name "$BACKUP_GLOB" -type f -printf "%T@ %p\n" | \
  sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$NEWEST_BACKUP" ]; then
  echo "Error: No backup files found in /backups"
  exit 1
fi

echo "Found newest backup: $(basename "$NEWEST_BACKUP")"

# Extract just the timestamp portion (for the extracted folder)
BACKUP_TIMESTAMP=$(basename "$NEWEST_BACKUP" .tar.gz | sed 's/acla_//')

# Create temp directory inside container
TEMPDIR="/tmp/mongorestore_$(date +%s)"
docker exec $CONTAINER_NAME mkdir -p "$TEMPDIR"

# Extract backup
echo "Extracting backup..."
docker exec $CONTAINER_NAME tar -xzvf "$NEWEST_BACKUP" -C "$TEMPDIR" || {
  echo "Error extracting backup"
  exit 1
}

# Restore to MongoDB
echo "Restoring to database '$MONGO_DATEBASE'..."
docker exec $CONTAINER_NAME mongorestore \
  --username "$MONGO_INITDB_ROOT_USERNAME" \
  --password "$MONGO_INITDB_ROOT_PASSWORD" \
  --authenticationDatabase admin \
  --db "$MONGO_DATEBASE" \
  --drop \
  "$TEMPDIR/$BACKUP_TIMESTAMP/$MONGO_DATEBASE" || {
    echo "Error during mongorestore"
    exit 1
}

# Cleanup
echo "Cleaning up..."
docker exec $CONTAINER_NAME rm -rf "$TEMPDIR"

echo "Successfully restored $(basename "$NEWEST_BACKUP") to $MONGO_DATEBASE"