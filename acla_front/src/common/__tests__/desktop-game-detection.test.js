const {
    detectSupportedDesktopGame,
    parseTasklistImageNames,
} = require('../../../public/desktop-game-detection');

describe('desktop game tasklist detection', () => {
    it('parses CSV and table tasklist output', () => {
        expect(parseTasklistImageNames([
            '"AC2-Win64-Shipping.exe","4120","Console","1","1,024 K"',
            'acs.exe                  9824 Console                    1    100,000 K',
        ].join('\r\n'))).toEqual(['ac2-win64-shipping.exe', 'acs.exe']);
    });

    it.each([
        ['acs.exe', 'ac'],
        ['ACS_X86.EXE', 'ac'],
        ['ac2-WIN64-shipping.EXE', 'acc'],
        ['IRACINGSIM64DX11.EXE', 'iracing'],
    ])('matches %s case-insensitively', (executable, expectedGame) => {
        const output = `"${executable}","4120","Console","1","1,024 K"`;
        expect(detectSupportedDesktopGame(output)).toBe(expectedGame);
    });

    it('returns null when no supported simulator is running', () => {
        expect(detectSupportedDesktopGame('"iRacingUI.exe","4120","Console","1","1,024 K"')).toBeNull();
    });

    it('uses ACC, AC, then iRacing priority when multiple games are running', () => {
        const allGames = [
            '"iRacingSim64DX11.exe","1","Console","1","1,024 K"',
            '"acs.exe","2","Console","1","1,024 K"',
            '"AC2-Win64-Shipping.exe","3","Console","1","1,024 K"',
        ].join('\r\n');
        const acAndIracing = [
            '"iRacingSim64DX11.exe","1","Console","1","1,024 K"',
            '"acs_x86.exe","2","Console","1","1,024 K"',
        ].join('\r\n');

        expect(detectSupportedDesktopGame(allGames)).toBe('acc');
        expect(detectSupportedDesktopGame(acAndIracing)).toBe('ac');
    });
});
