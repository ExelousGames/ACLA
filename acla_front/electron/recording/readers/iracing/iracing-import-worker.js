'use strict';

const fs = require('fs');
const { parentPort, workerData } = require('worker_threads');
const { readLiveRecordingInfo, findMatchingIBTFiles, convertIBTFiles } = require('./iracing-ibt');

async function prepare() {
  const { liveFilePath, telemetryDirectory } = workerData;
  const outputPath = `${liveFilePath}.iracing-recorded.jsonl`;
  // A completed conversion is immutable and can be reused when an upload is retried.
  try {
    const converted = await fs.promises.stat(outputPath);
    const liveStat = await fs.promises.stat(liveFilePath);
    if (converted.size > 0 && converted.mtimeMs >= liveStat.mtimeMs) return { filePath: outputPath };
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const live = await readLiveRecordingInfo(liveFilePath);
  let filePaths = await findMatchingIBTFiles(telemetryDirectory, live);
  if (!filePaths.length) {
    filePaths = await new Promise((resolve) => {
      parentPort.once('message', (message) => resolve(message.filePaths));
      parentPort.postMessage({ type: 'select-files' });
    });
  }
  if (!filePaths?.length) throw new Error('Select the session’s .ibt file to upload both versions. Enable disk telemetry in iRacing while driving if no file exists.');
  return convertIBTFiles(filePaths, outputPath, live);
}

prepare().then((result) => parentPort.postMessage({ type: 'complete', ...result }))
  .catch((error) => parentPort.postMessage({ type: 'error', message: error.message }))
  .finally(() => parentPort.close());
