'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { TextDecoder } = require('util');
const { IRacingIBTAdapter, IRACING_IBT_VARIABLES } = require('./iracing-ibt-adapter');

// IRSDK header / disk subheader / variable descriptors:
// https://github.com/kutu/pyirsdk/blob/master/irsdk.py
const TYPE_SIZES = [1, 1, 4, 4, 4, 8];
const WANTED_VARIABLES = new Set(IRACING_IBT_VARIABLES);
const decodeString = (buffer, encoding = 'windows-1252') => (
  new TextDecoder(encoding).decode(buffer.subarray(0, buffer.indexOf(0) < 0 ? buffer.length : buffer.indexOf(0)))
);

async function readExactly(file, size, position) {
  const buffer = Buffer.alloc(size);
  let offset = 0;
  while (offset < size) {
    const { bytesRead } = await file.read(buffer, offset, size - offset, position + offset);
    if (!bytesRead) throw new Error('The .ibt file is incomplete. Exit the car in iRacing and retry the upload.');
    offset += bytesRead;
  }
  return buffer;
}

class IBTFile {
  static async open(filePath) {
    if (path.extname(filePath).toLowerCase() !== '.ibt') throw new Error('Select an iRacing .ibt telemetry file.');
    const file = await fs.promises.open(filePath, 'r');
    try {
      const stat = await file.stat();
      if (!stat.isFile()) throw new Error('The .ibt path must be a regular file.');
      const header = await readExactly(file, 144, 0);
      const sessionLength = header.readInt32LE(16);
      const sessionOffset = header.readInt32LE(20);
      const variableCount = header.readInt32LE(24);
      const variableOffset = header.readInt32LE(28);
      const rowLength = header.readInt32LE(36);
      const firstTick = header.readInt32LE(48);
      const dataOffset = header.readInt32LE(52);
      const rowCount = header.readInt32LE(140);
      const rangeValid = (offset, size) => offset >= 144 && size > 0 && offset + size <= stat.size;
      if (header.readInt32LE(0) < 1 || header.readInt32LE(8) <= 0
        || sessionLength > 4 * 1024 * 1024 || variableCount < 1 || variableCount > 10000
        || rowLength < 1 || rowLength > 16 * 1024 * 1024
        || !rangeValid(sessionOffset, sessionLength) || !rangeValid(variableOffset, variableCount * 144)) {
        throw new Error('Invalid iRacing .ibt header.');
      }
      if (rowCount < 1 || !rangeValid(dataOffset, rowCount * rowLength)) {
        throw new Error('The .ibt file has no finalized telemetry. Exit the car in iRacing and retry the upload.');
      }
      const sessionBytes = await readExactly(file, sessionLength, sessionOffset);
      const sessionInfo = decodeString(sessionBytes, /Encoding:\s*UTF8/.test(sessionBytes.toString('ascii')) ? 'utf-8' : 'windows-1252');
      const table = await readExactly(file, variableCount * 144, variableOffset);
      const variables = [];
      for (let i = 0; i < variableCount; i += 1) {
        const base = i * 144;
        const type = table.readInt32LE(base);
        const offset = table.readInt32LE(base + 4);
        const count = table.readInt32LE(base + 8);
        const countAsTime = table[base + 12] !== 0;
        const name = decodeString(table.subarray(base + 16, base + 48));
        const unit = decodeString(table.subarray(base + 112, base + 144));
        if (!TYPE_SIZES[type] || count < 1 || offset < 0 || offset + count * TYPE_SIZES[type] > rowLength) {
          throw new Error(`Invalid .ibt variable: ${name}.`);
        }
        if (WANTED_VARIABLES.has(name)) variables.push({ name, type, offset, count, countAsTime, unit });
      }
      if (!variables.some(({ name }) => name === 'Speed')) throw new Error('The .ibt file has no vehicle telemetry.');
      const adapter = new IRacingIBTAdapter(variables);
      adapter.updateSession(sessionInfo);
      adapter.buildStaticFields(adapter.session.DriverInfo?.DriverCarIdx);
      return Object.assign(new IBTFile(), { file, filePath, stat, rowLength, rowCount, firstTick, dataOffset, variables, adapter });
    } catch (error) {
      await file.close();
      throw error;
    }
  }

  decodeRow(row, index) {
    const values = {};
    for (const { name, type, offset, count, countAsTime } of this.variables) {
      const entries = [];
      if (type === 0) {
        values[name] = decodeString(row.subarray(offset, offset + count));
        continue;
      }
      for (let i = 0; i < count; i += 1) {
        const position = offset + i * TYPE_SIZES[type];
        entries.push(type === 1 ? row[position] !== 0 : type === 2 ? row.readInt32LE(position)
          : type === 3 ? row.readUInt32LE(position) : type === 4 ? row.readFloatLE(position) : row.readDoubleLE(position));
      }
      if (entries.every((value) => typeof value === 'boolean' || Number.isFinite(value))) {
        values[name] = count === 1 || countAsTime ? entries[entries.length - 1] : entries;
      }
    }
    return this.adapter.adapt({ type: 'sample', tick: values.SessionTick ?? this.firstTick + index, values });
  }

  async *rows() {
    for (let index = 0; index < this.rowCount;) {
      const count = Math.min(250, this.rowCount - index);
      const buffer = await readExactly(this.file, count * this.rowLength, this.dataOffset + index * this.rowLength);
      for (let i = 0; i < count; i += 1) {
        yield this.decodeRow(buffer.subarray(i * this.rowLength, (i + 1) * this.rowLength), index + i);
      }
      index += count;
    }
    const after = await this.file.stat();
    if (after.size !== this.stat.size || after.mtimeMs !== this.stat.mtimeMs) {
      throw new Error('iRacing is still writing the .ibt file. Exit the car and retry the upload.');
    }
  }

  close() { return this.file.close(); }
}

async function readLiveRecordingInfo(filePath) {
  const stat = await fs.promises.stat(filePath);
  const stream = fs.createReadStream(filePath);
  const lines = readline.createInterface({ input: stream, crlfDelay: Infinity });
  let sample;
  try {
    for await (const line of lines) {
      if (!line.trim()) continue;
      const row = JSON.parse(line);
      if (row.Static_track && row.Static_car_model) { sample = row; break; }
    }
  } finally { lines.close(); stream.destroy(); }
  if (!sample) throw new Error('The live recording has no track and car information to match an .ibt file.');
  return { sample, startedAt: stat.birthtimeMs, endedAt: stat.mtimeMs };
}

function matchesVehicle(ibt, live) {
  const sample = ibt.adapter.staticFields;
  return ['Static_track', 'Static_car_model', 'Static_player_name'].every((field) => (
    !live.sample[field] || sample[field] === live.sample[field]
  ));
}

function sessionKey(ibt) {
  const weekend = ibt.adapter.session.WeekendInfo || {};
  return JSON.stringify([weekend.SessionID, weekend.SubSessionID, ibt.adapter.session.DriverInfo?.DriverCarIdx]);
}

async function findMatchingIBTFiles(directory, live) {
  let entries;
  try { entries = await fs.promises.readdir(directory, { withFileTypes: true }); }
  catch (error) { if (error.code === 'ENOENT') return []; throw error; }
  const matches = [];
  for (const entry of entries) {
    if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== '.ibt') continue;
    const filePath = path.join(directory, entry.name);
    const stat = await fs.promises.stat(filePath);
    if (stat.mtimeMs < live.startedAt - 2000 || stat.birthtimeMs > live.endedAt + 2000) continue;
    let ibt;
    try {
      ibt = await IBTFile.open(filePath);
      if (matchesVehicle(ibt, live)) matches.push({ filePath, key: sessionKey(ibt), createdAt: stat.birthtimeMs });
    } catch { /* Unfinalized or unreadable files can be selected after the driver exits the car. */ }
    finally { await ibt?.close(); }
  }
  // A second simulator session in the same time window needs explicit selection.
  if (new Set(matches.map(({ key }) => key)).size > 1) return [];
  return matches.sort((a, b) => a.createdAt - b.createdAt).map(({ filePath }) => filePath);
}

async function convertIBTFiles(filePaths, outputPath, live) {
  const pendingPath = `${outputPath}.partial`;
  // Recover an interrupted conversion; the main process allows only one import.
  await fs.promises.unlink(pendingPath).catch((error) => { if (error.code !== 'ENOENT') throw error; });
  const output = await fs.promises.open(pendingPath, 'wx');
  let rowCount = 0;
  let firstSessionKey;
  try {
    const orderedFiles = await Promise.all([...new Set(filePaths)].map(async (filePath) => ({
      filePath, stat: await fs.promises.stat(filePath),
    })));
    orderedFiles.sort((a, b) => a.stat.birthtimeMs - b.stat.birthtimeMs || a.filePath.localeCompare(b.filePath));
    for (const { filePath } of orderedFiles) {
      const ibt = await IBTFile.open(filePath);
      try {
        if (!matchesVehicle(ibt, live)) throw new Error('The selected .ibt file does not match the live recording’s track, car and driver.');
        const key = sessionKey(ibt);
        if (firstSessionKey !== undefined && key !== firstSessionKey) throw new Error('Select .ibt files from the same iRacing session.');
        firstSessionKey = key;
        let batch = [];
        for await (const row of ibt.rows()) {
          batch.push(JSON.stringify(row) + '\n');
          rowCount += 1;
          if (batch.length === 250) { await output.writeFile(batch.join('')); batch = []; }
        }
        if (batch.length) await output.writeFile(batch.join(''));
      } finally { await ibt.close(); }
    }
    if (!rowCount) throw new Error('No .ibt telemetry data found to upload.');
    await output.close();
    await fs.promises.rename(pendingPath, outputPath);
    return { filePath: outputPath, rowCount };
  } catch (error) {
    await output.close();
    await fs.promises.unlink(pendingPath).catch(() => undefined);
    throw error;
  }
}

module.exports = { IBTFile, readLiveRecordingInfo, findMatchingIBTFiles, convertIBTFiles };
