'use strict';

const { parentPort, workerData } = require('worker_threads');
const { IBTFile, convertIBTFiles } = require('./iracing-ibt');

async function importFile() {
  const { sourcePath, outputPath } = workerData;
  const ibt = await IBTFile.open(sourcePath);
  let sample;
  try {
    sample = ibt.adapter.staticFields;
  } finally {
    await ibt.close();
  }
  const result = await convertIBTFiles([sourcePath], outputPath, { sample });
  return { ...result, track: sample.Static_track || '', car: sample.Static_car_model || '' };
}

importFile().then((result) => parentPort.postMessage({ type: 'complete', ...result }))
  .catch((error) => parentPort.postMessage({ type: 'error', message: error.message }))
  .finally(() => parentPort.close());
