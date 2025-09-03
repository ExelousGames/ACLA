# Chunked Download API for Racing Sessions

This document describes the new chunked download API endpoints for retrieving all racing sessions from the database securely and efficiently.

## Overview

The chunked download system allows clients to download large racing session datasets in manageable pieces, preventing memory overflow and network timeouts. The system uses a three-step process:

1. **Initialize Download** - Get metadata and download ID
2. **Download Chunks** - Retrieve data in chunks
3. **Check Status** - Monitor download progress (optional)

## API Endpoints

### 1. Initialize Download

**Endpoint**: `POST /racing-session/download/init`

**Headers**: 
- `Authorization: Bearer <JWT_TOKEN>`

**Request Body**:
```json
{
  "userId": "optional_user_id_to_filter",
  "chunkSize": 1000
}
```

**Response**:
```json
{
  "downloadId": "uuid-string",
  "totalSessions": 25,
  "totalChunks": 150,
  "sessionMetadata": [
    {
      "sessionId": "session_id_1",
      "session_name": "Practice Session 1",
      "map": "Monza",
      "car_name": "Ferrari 488 GT3",
      "userId": "user123",
      "dataSize": 5000,
      "chunkCount": 5
    }
    // ... more sessions
  ]
}
```

### 2. Download Chunk

**Endpoint**: `POST /racing-session/download/chunk`

**Headers**: 
- `Authorization: Bearer <JWT_TOKEN>`

**Request Body**:
```json
{
  "downloadId": "uuid-from-init-response",
  "sessionId": "session_id_from_metadata",
  "chunkIndex": 0
}
```

**Response**:
```json
{
  "downloadId": "uuid-string",
  "sessionId": "session_id_1",
  "chunkIndex": 0,
  "totalChunks": 5,
  "data": [
    // Array of telemetry data points (up to chunkSize items)
  ],
  "isComplete": false
}
```

### 3. Check Download Status

**Endpoint**: `POST /racing-session/download/status`

**Headers**: 
- `Authorization: Bearer <JWT_TOKEN>`

**Request Body**:
```json
{
  "downloadId": "uuid-from-init-response"
}
```

**Response**:
```json
{
  "downloadId": "uuid-string",
  "totalSessions": 25,
  "totalChunks": 150,
  "downloadedChunks": 75,
  "progress": 50.0,
  "isComplete": false,
  "createdAt": "2025-09-03T10:00:00.000Z"
}
```

## Usage Example (JavaScript)

```javascript
class RacingSessionDownloader {
  constructor(baseUrl, authToken) {
    this.baseUrl = baseUrl;
    this.authToken = authToken;
    this.headers = {
      'Authorization': `Bearer ${authToken}`,
      'Content-Type': 'application/json'
    };
  }

  async downloadAllSessions(userId = null, chunkSize = 1000) {
    try {
      // Step 1: Initialize download
      const initResponse = await fetch(`${this.baseUrl}/racing-session/download/init`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({ userId, chunkSize })
      });
      
      const initData = await initResponse.json();
      console.log(`Starting download of ${initData.totalSessions} sessions in ${initData.totalChunks} chunks`);

      const allSessionsData = {};

      // Step 2: Download all chunks for all sessions
      for (const sessionMeta of initData.sessionMetadata) {
        console.log(`Downloading session: ${sessionMeta.session_name}`);
        allSessionsData[sessionMeta.sessionId] = {
          metadata: sessionMeta,
          data: []
        };

        // Download all chunks for this session
        for (let chunkIndex = 0; chunkIndex < sessionMeta.chunkCount; chunkIndex++) {
          const chunkResponse = await fetch(`${this.baseUrl}/racing-session/download/chunk`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
              downloadId: initData.downloadId,
              sessionId: sessionMeta.sessionId,
              chunkIndex
            })
          });

          const chunkData = await chunkResponse.json();
          allSessionsData[sessionMeta.sessionId].data.push(...chunkData.data);

          console.log(`Downloaded chunk ${chunkIndex + 1}/${sessionMeta.chunkCount} for session ${sessionMeta.session_name}`);
        }
      }

      console.log('Download completed successfully');
      return allSessionsData;

    } catch (error) {
      console.error('Download failed:', error);
      throw error;
    }
  }

  async checkDownloadProgress(downloadId) {
    const response = await fetch(`${this.baseUrl}/racing-session/download/status`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ downloadId })
    });
    
    return await response.json();
  }
}

// Usage
const downloader = new RacingSessionDownloader('http://localhost:3000', 'your-jwt-token');
const allSessions = await downloader.downloadAllSessions();
```

## Security Features

1. **JWT Authentication**: All endpoints require valid JWT tokens
2. **Download Session Validation**: Each chunk request is validated against the initialized download session
3. **Session Expiration**: Download sessions expire after 1 hour for security
4. **User Filtering**: Optional user-based filtering to restrict access to specific user data
5. **State Tracking**: Server tracks which chunks have been downloaded to prevent data inconsistencies

## Performance Considerations

1. **Chunk Size**: Default chunk size is 1000 items. Adjust based on your data size and network conditions
2. **Memory Management**: Server manages memory efficiently by processing chunks on-demand
3. **Cleanup**: Old download sessions are automatically cleaned up to prevent memory leaks
4. **Progress Tracking**: Built-in progress tracking helps with user experience and debugging

## Error Handling

Common error responses:
- `400 Bad Request`: Invalid download ID, session not found, or expired download session
- `401 Unauthorized`: Invalid or missing JWT token
- `500 Internal Server Error`: Database or processing errors

Each error response includes a descriptive message to help with troubleshooting.
