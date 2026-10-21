const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const prepareVjoy = require('../prepare-vjoy');

async function fixture(t) {
    const projectDir = await fs.mkdtemp(path.join(os.tmpdir(), 'acla-vjoy-test-'));
    t.after(() => fs.rm(projectDir, { recursive: true, force: true }));
    return { electronPlatformName: 'win32', arch: 1, packager: { projectDir } };
}

test('non-Windows packaging does not download or require vJoy', async () => {
    await prepareVjoy({ electronPlatformName: 'darwin' }, () => assert.fail('unexpected download'));
    await prepareVjoy({ electronPlatformName: 'linux' }, () => assert.fail('unexpected download'));
});

test('unsupported Windows architectures fail before downloading a driver', async () => {
    for (const arch of [0, 2, 3]) {
        await assert.rejects(prepareVjoy({ electronPlatformName: 'win32', arch }, () => assert.fail('unexpected download')), /Windows x64/);
    }
});

test('a modified cached installer stops packaging without a network fallback', async (t) => {
    const context = await fixture(t);
    const directory = path.join(context.packager.projectDir, 'assets/vjoy');
    await fs.mkdir(directory, { recursive: true });
    await fs.writeFile(path.join(directory, 'vJoySetup.exe'), 'modified installer');
    await assert.rejects(prepareVjoy(context, () => assert.fail('unexpected download')), /checksum mismatch/);
});

test('failed HTTP responses do not create an installer', async (t) => {
    const context = await fixture(t);
    await assert.rejects(prepareVjoy(context, async (url) => {
        assert.equal(url, prepareVjoy.release.url);
        return { ok: false, status: 404 };
    }), /HTTP 404/);
    await assert.rejects(fs.access(path.join(context.packager.projectDir, 'assets/vjoy/vJoySetup.exe')), { code: 'ENOENT' });
});

test('a corrupted download is never published for packaging', async (t) => {
    const context = await fixture(t);
    await assert.rejects(prepareVjoy(context, async () => ({
        ok: true, arrayBuffer: async () => Buffer.from('not the signed release'),
    })), /checksum mismatch/);
    await assert.rejects(fs.access(path.join(context.packager.projectDir, 'assets/vjoy/vJoySetup.exe')), { code: 'ENOENT' });
});
