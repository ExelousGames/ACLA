// A source selection grants one display-media request from the main frame only.
function registerScreenCapture({ ipcMain, desktopCapturer, getMainWindow }) {
  let selection = null;
  const owner = (event) => {
    const window = getMainWindow();
    if (!window || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== window.webContents.mainFrame) {
      throw new Error('Screen capture is available only in the main workspace.');
    }
    return window.webContents;
  };
  const getSources = () => desktopCapturer.getSources({
    types: ['window', 'screen'], thumbnailSize: { width: 0, height: 0 },
  });

  ipcMain.handle('screen-capture-sources', async (event) => {
    owner(event);
    return (await getSources()).map(({ id, name }) => ({ id, name }));
  });
  ipcMain.handle('screen-capture-select', async (event, id) => {
    const contents = owner(event);
    const frame = contents.mainFrame;
    selection = null;
    if (typeof id !== 'string' || !(await getSources()).some((source) => source.id === id)) {
      throw new Error('The selected window is no longer available. Refresh the source list.');
    }
    owner(event);
    if (contents.mainFrame !== frame) throw new Error('The workspace navigated during screen selection.');
    selection = { id, frame, expiresAt: Date.now() + 5000 };
  });

  return (window) => {
    selection = null;
    window.webContents.on('did-start-navigation', () => { selection = null; });
    window.webContents.session.setDisplayMediaRequestHandler(async (request, callback) => {
      const chosen = selection;
      selection = null;
      if (!chosen || chosen.expiresAt < Date.now() || request.frame !== chosen.frame
        || request.frame !== getMainWindow()?.webContents.mainFrame || !request.videoRequested) {
        callback({});
        return;
      }
      try {
        const source = (await getSources()).find(({ id }) => id === chosen.id);
        const window = getMainWindow();
        const stillCurrent = window && !window.isDestroyed() && request.frame === window.webContents.mainFrame;
        callback(source && stillCurrent ? { video: source } : {});
      } catch {
        callback({});
      }
    });
  };
}

module.exports = { registerScreenCapture };
