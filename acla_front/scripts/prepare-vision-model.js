const fs = require('fs/promises');
const path = require('path');
const https = require('https');
const { createHash } = require('crypto');
const model = require('../public/vision-models/yolop.json');

function download(url, redirects = 0) {
    return new Promise((resolve, reject) => {
        const request = https.get(url, (response) => {
            if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location && redirects < 5) {
                response.resume();
                resolve(download(new URL(response.headers.location, url).href, redirects + 1));
                return;
            }
            if (response.statusCode !== 200) {
                response.resume();
                reject(new Error(`Model download returned HTTP ${response.statusCode}.`));
                return;
            }
            const chunks = [];
            let size = 0;
            response.on('data', (chunk) => {
                size += chunk.length;
                if (size > model.bytes) response.destroy(new Error('Model download exceeds the expected size.'));
                else chunks.push(chunk);
            });
            response.on('end', () => resolve(Buffer.concat(chunks)));
            response.on('error', reject);
        });
        request.setTimeout(60000, () => request.destroy(new Error('Model download timed out.')));
        request.on('error', reject);
    });
}

function isVerified(bytes) {
    return bytes.length === model.bytes && createHash('sha256').update(bytes).digest('hex') === model.sha256;
}

async function prepareVisionModel() {
    const target = path.resolve(__dirname, '../public/vision-models', model.file);
    try {
        if (isVerified(await fs.readFile(target))) {
            console.log('Verified bundled YOLOP road-segmentation weights.');
            return;
        }
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    console.log('Downloading pretrained YOLOP road-segmentation weights (32 MB, once per checkout)…');
    const bytes = await download(model.url);
    if (!isVerified(bytes)) throw new Error('YOLOP model checksum mismatch. Download was not installed.');
    await fs.mkdir(path.dirname(target), { recursive: true });
    const temporary = `${target}.${process.pid}.tmp`;
    await fs.writeFile(temporary, bytes);
    await fs.rename(temporary, target);
    console.log('Prepared bundled YOLOP road-segmentation model.');
}

module.exports = { prepareVisionModel };
if (require.main === module) {
    prepareVisionModel().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
