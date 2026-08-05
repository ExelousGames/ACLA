const SUPPORTED_GAME_EXECUTABLES = {
  acc: ['ac2-win64-shipping.exe'],
  ac: ['acs.exe', 'acs_x86.exe'],
  iracing: ['iracingsim64dx11.exe'],
};

const GAME_PRIORITY = ['acc', 'ac', 'iracing'];

function parseTasklistImageNames(output) {
  if (typeof output !== 'string') {
    return [];
  }

  return output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const csvImageName = line.match(/^"((?:[^"]|"")*)"/);
      const imageName = csvImageName
        ? csvImageName[1].replace(/""/g, '"')
        : line.split(/\s+/)[0];

      return imageName.toLowerCase();
    });
}

function detectSupportedDesktopGame(output) {
  const runningExecutables = new Set(parseTasklistImageNames(output));

  for (const game of GAME_PRIORITY) {
    if (SUPPORTED_GAME_EXECUTABLES[game].some((executable) => runningExecutables.has(executable))) {
      return game;
    }
  }

  return null;
}

module.exports = {
  detectSupportedDesktopGame,
  parseTasklistImageNames,
};
