import React from 'react';
import { render, screen, within } from '@testing-library/react';
import { buildRoadsideLabelGeometry, DriverExpertComparisonRoadSigns } from './DriverExpertComparisonRoadSigns';

const project = (point: { x: number; y: number } | undefined) => (
    point ? { ...point, svgX: point.x, svgY: point.y } : undefined
);
const positionAt = (index: number) => project({ x: 60 + (index - 100) * 20, y: 180 })!;
const stream = [100, 110, 120, 130, 150].map((sourceIndex) => ({
    sourceIndex, trajectory: positionAt(sourceIndex),
}));
const label = { label: 'Late braking', startIndex: 104, endIndex: 116, category: 'mistakes' as const };
const recovery = { label: 'Smooth recovery', startIndex: 112, endIndex: 120, category: 'recovery' as const };
const camera = { project: (point: ReturnType<typeof project>) => point };
const props = {
    ranges: buildRoadsideLabelGeometry([label, recovery], stream, project),
    camera,
    viewportWidth: 760,
    viewportHeight: 300,
};
const signAt = (sourceIndex: number) => <svg><DriverExpertComparisonRoadSigns {...props}
    sourceIndex={sourceIndex} driverPosition={positionAt(sourceIndex)} /></svg>;
const signsInState = (state: string) => screen.queryAllByTestId('comparison-road-sign')
    .filter((sign) => sign.getAttribute('data-state') === state);
const expectAnchor = (sign: HTMLElement, x: number, y = 180) => {
    expect(Number(sign.getAttribute('data-anchor-x'))).toBeCloseTo(x);
    expect(Number(sign.getAttribute('data-anchor-y'))).toBeCloseTo(y);
    const post = within(sign).getByTestId('comparison-label-sign-post');
    const coordinates = post.getAttribute('d')!.match(/-?[\d.]+/g)!.map(Number);
    expect(coordinates.slice(0, 2)).toEqual([x, y]);
};

describe('road signs on the player trajectory', () => {
    it.each([1, 1.6])('keeps sign dimensions proportional to viewport scale %s', (scale) => {
        render(<svg><DriverExpertComparisonRoadSigns {...props} ranges={[props.ranges[0]]}
            sourceIndex={110} driverPosition={positionAt(110)} scale={scale} /></svg>);
        const board = screen.getByTestId('comparison-label-sign-board');
        expect(Number(board.getAttribute('width'))).toBeCloseTo(160 * scale);
        expect(Number(board.getAttribute('height'))).toBeCloseTo(44 * scale);
        expect(parseFloat(screen.getByTestId('comparison-label-sign').querySelector('text')!.style.fontSize))
            .toBeCloseTo(10 * scale);
    });

    it('scales the entire sign with camera depth and hides it behind the near plane', () => {
        const atDepth = (perspectiveScale: number) => <svg><DriverExpertComparisonRoadSigns {...props}
            ranges={[props.ranges[0]]} sourceIndex={110} driverPosition={positionAt(110)}
            camera={{ project: (point) => point ? { ...point, perspectiveScale } : undefined }} /></svg>;
        const view = render(atDepth(1));
        const board = screen.getByTestId('comparison-label-sign-board');
        const originalWidth = Number(board.getAttribute('width'));
        const originalHeight = Number(board.getAttribute('height'));
        view.rerender(atDepth(0.5));
        expect(Number(board.getAttribute('width'))).toBeCloseTo(originalWidth / 2);
        expect(Number(board.getAttribute('height'))).toBeCloseTo(originalHeight / 2);
        expectAnchor(signsInState('following')[0], 260);
        view.rerender(atDepth(2));
        expect(Number(board.getAttribute('width'))).toBeCloseTo(originalWidth * 2);
        expect(Number(board.getAttribute('height'))).toBeCloseTo(originalHeight * 2);
        view.rerender(atDepth(0));
        expect(screen.queryByTestId('comparison-road-signs')).not.toBeInTheDocument();
    });

    it('fades distant signs in as the driver approaches and keeps active signs fully visible', () => {
        const approaching = (distance: number) => <svg><DriverExpertComparisonRoadSigns {...props}
            ranges={[props.ranges[0]]} sourceIndex={100}
            driverPosition={project({ x: props.ranges[0].start!.x - distance, y: 180 })} /></svg>;
        const view = render(approaching(300));
        const opacity = () => Number(screen.getByTestId('comparison-road-sign').style.opacity);
        expect(opacity()).toBe(0);
        view.rerender(approaching(150));
        const distantOpacity = opacity();
        expect(distantOpacity).toBeGreaterThan(0);
        expect(distantOpacity).toBeLessThan(0.2);
        view.rerender(approaching(60));
        expect(opacity()).toBeGreaterThan(distantOpacity);
        expect(opacity()).toBeLessThan(1);
        view.rerender(approaching(0));
        expect(opacity()).toBe(1);
        view.rerender(signAt(113));
        expect(signsInState('following')[0]).toHaveStyle({ opacity: '1' });
    });

    it('fades released signs much faster than upcoming signs and resets on replay', () => {
        const atDistance = (released: boolean, distance: number) => <svg><DriverExpertComparisonRoadSigns {...props}
            ranges={[props.ranges[0]]} sourceIndex={released ? 120 : 100}
            driverPosition={project({
                x: released ? props.ranges[0].end!.x + distance : props.ranges[0].start!.x - distance,
                y: 180,
            })} /></svg>;
        const view = render(atDistance(false, 20));
        const opacity = () => Number(screen.getByTestId('comparison-road-sign').style.opacity);
        const upcomingOpacity = opacity();
        view.rerender(atDistance(true, 0));
        expect(opacity()).toBe(1);
        view.rerender(atDistance(true, 4));
        const justReleasedOpacity = opacity();
        expect(justReleasedOpacity).toBeGreaterThan(0);
        expect(justReleasedOpacity).toBeLessThan(1);
        view.rerender(atDistance(true, 12));
        expect(opacity()).toBeLessThan(justReleasedOpacity);
        expect(opacity()).toBeLessThan(upcomingOpacity / 2);
        view.rerender(atDistance(true, 20));
        expect(opacity()).toBe(0);
        view.rerender(atDistance(false, 20));
        expect(opacity()).toBe(upcomingOpacity);
    });

    it('uses world distance on both axes so camera changes do not change sign opacity', () => {
        const atPosition = (x: number, y: number, zoom = 1) => <svg><DriverExpertComparisonRoadSigns {...props}
            ranges={[props.ranges[0]]} sourceIndex={100} driverPosition={project({ x, y })}
            camera={{ project: (point) => point ? {
                ...point, svgX: point.svgX * zoom, svgY: point.svgY * zoom, perspectiveScale: zoom,
            } : undefined }} /></svg>;
        const view = render(atPosition(80, 180));
        const opacity = () => Number(screen.getByTestId('comparison-road-sign').style.opacity);
        const horizontalOpacity = opacity();
        expect(horizontalOpacity).toBeGreaterThan(0);
        expect(horizontalOpacity).toBeLessThan(1);
        view.rerender(atPosition(140, 120));
        expect(opacity()).toBeCloseTo(horizontalOpacity);
        view.rerender(atPosition(140, 120, 0.5));
        expect(opacity()).toBeCloseTo(horizontalOpacity);
    });

    it('waits at each start, follows the player, merges, and leaves each sign at its end', () => {
        const view = render(signAt(100));
        expect(signsInState('waiting')).toHaveLength(2);
        expectAnchor(signsInState('waiting').find((sign) => sign.getAttribute('aria-label') === 'Late braking')!, 140);
        expectAnchor(signsInState('waiting').find((sign) => sign.getAttribute('aria-label') === 'Smooth recovery')!, 300);

        view.rerender(signAt(103.99));
        expect(signsInState('following')).toHaveLength(0);
        view.rerender(signAt(104));
        expect(signsInState('following')[0]).toHaveAttribute('aria-label', 'Late braking');
        expectAnchor(signsInState('following')[0], 140);

        view.rerender(signAt(110));
        expectAnchor(signsInState('following')[0], 260);
        expectAnchor(signsInState('waiting')[0], 300);

        view.rerender(signAt(112));
        expect(screen.getAllByTestId('comparison-label-sign')).toHaveLength(1);
        const merged = signsInState('following')[0];
        expect(merged).toHaveAttribute('aria-label', 'Late braking, Smooth recovery');
        expectAnchor(merged, 300);
        const mergedHeight = Number(within(merged).getByTestId('comparison-label-sign-board').getAttribute('height'));
        expect(screen.queryByTestId('comparison-label-range')).not.toBeInTheDocument();
        expect(merged).not.toHaveTextContent(/104|116|112|120|→/);

        view.rerender(signAt(116));
        expect(signsInState('following')[0]).toHaveAttribute('aria-label', 'Smooth recovery');
        expect(signsInState('released')[0]).toHaveAttribute('aria-label', 'Late braking');
        expect(Number(within(signsInState('following')[0]).getByTestId('comparison-label-sign-board')
            .getAttribute('height'))).toBeLessThan(mergedHeight);

        view.rerender(signAt(118));
        expectAnchor(signsInState('following')[0], 420);
        expectAnchor(signsInState('released')[0], 380);
        view.rerender(signAt(120));
        expect(signsInState('following')).toHaveLength(0);
        expect(signsInState('released')).toHaveLength(2);
        view.rerender(signAt(125));
        expectAnchor(signsInState('released').find((sign) => sign.getAttribute('aria-label') === 'Late braking')!, 380);
        expectAnchor(signsInState('released').find((sign) => sign.getAttribute('aria-label') === 'Smooth recovery')!, 460);

        view.rerender(signAt(100));
        expect(signsInState('waiting')).toHaveLength(2);
        expect(signsInState('released')).toHaveLength(0);
    });

    it('keeps released signs fixed in the world while the camera and player move on', () => {
        const withCamera = (sourceIndex: number, pan: number) => <svg><DriverExpertComparisonRoadSigns {...props}
            sourceIndex={sourceIndex} driverPosition={positionAt(sourceIndex)}
            camera={{ project: (point) => point ? { ...point, svgX: point.svgX - pan } : undefined }} /></svg>;
        const view = render(withCamera(117, 0));
        expectAnchor(signsInState('released')[0], 380);
        expectAnchor(signsInState('following')[0], 400);
        view.rerender(withCamera(119, 40));
        expectAnchor(signsInState('released')[0], 340);
        expect(signsInState('released')[0]).toHaveAttribute('data-world-x', '380');
        expectAnchor(signsInState('following')[0], 400);
        view.rerender(withCamera(140, 500));
        expect(screen.queryByTestId('comparison-road-signs')).not.toBeInTheDocument();
    });

    it('deduplicates active text but releases each interval independently', () => {
        const ranges = buildRoadsideLabelGeometry([label, { ...label, startIndex: 112, endIndex: 120 }], stream, project);
        const sign = (sourceIndex: number) => <svg><DriverExpertComparisonRoadSigns {...props} ranges={ranges}
            sourceIndex={sourceIndex} driverPosition={positionAt(sourceIndex)} /></svg>;
        const view = render(sign(113));
        expect(screen.getByTestId('comparison-label-sign').querySelectorAll('text')).toHaveLength(1);
        view.rerender(sign(118));
        expect(signsInState('following')[0]).toHaveAttribute('aria-label', 'Late braking');
        expectAnchor(signsInState('following')[0], 420);
        expectAnchor(signsInState('released')[0], 380);
        view.rerender(sign(120));
        expect(signsInState('released')).toHaveLength(2);
        expect(signsInState('following')).toHaveLength(0);
    });

    it('merges only active labels, including simultaneous, nested, and adjacent intervals', () => {
        const ranges = buildRoadsideLabelGeometry([
            { ...label, label: 'Outer', startIndex: 100, endIndex: 130 },
            { ...recovery, label: 'Inner', startIndex: 110, endIndex: 120 },
            { ...label, label: 'Brief', startIndex: 110, endIndex: 115 },
            { ...label, label: 'Next', startIndex: 130, endIndex: 140 },
        ], stream, project);
        const sign = (sourceIndex: number) => <svg><DriverExpertComparisonRoadSigns {...props} ranges={ranges}
            sourceIndex={sourceIndex} driverPosition={positionAt(sourceIndex)} /></svg>;
        const view = render(sign(110));
        expect(signsInState('following')).toHaveLength(1);
        expect(signsInState('following')[0]).toHaveAttribute('aria-label', 'Outer, Inner, Brief');
        expect(signsInState('waiting')[0]).toHaveAttribute('aria-label', 'Next');
        view.rerender(sign(120));
        expect(signsInState('following')[0]).toHaveAttribute('aria-label', 'Outer');
        expect(signsInState('released')).toHaveLength(2);
        view.rerender(sign(130));
        expect(signsInState('following')[0]).toHaveAttribute('aria-label', 'Next');
        expect(signsInState('released')).toHaveLength(3);
    });

    it('interpolates original source-index boundaries and clips them to the available trajectory', () => {
        const [geometry] = buildRoadsideLabelGeometry([label], stream, project);
        expect(geometry.start).toEqual(positionAt(104));
        expect(geometry.end).toEqual(positionAt(116));
        const clipped = buildRoadsideLabelGeometry([
            { ...label, startIndex: 90, endIndex: 105 },
            { ...label, startIndex: 145, endIndex: 160 },
            { ...label, startIndex: 150, endIndex: 151 },
            { ...label, startIndex: 90, endIndex: 100 },
            { ...label, startIndex: 151, endIndex: 160 },
        ], stream, project);
        expect(clipped.map((range) => [range.start?.x, range.end?.x]))
            .toEqual([[60, 160], [960, 1060], [1060, 1060]]);
        expect(buildRoadsideLabelGeometry([label], [{ trajectory: { x: 1, y: 2 } }], project)).toEqual([]);
        const [missing] = buildRoadsideLabelGeometry([label], [stream[0], { sourceIndex: 110 }, stream[2]], project);
        expect(missing.start).toBeUndefined();
        expect(missing.end).toBeUndefined();
    });

    it('keeps overlapping released and following signs at the same offset from their anchors', () => {
        const ranges = props.ranges.map((range) => ({ ...range, end: project({ x: 380, y: 200 }) }));
        render(<svg><DriverExpertComparisonRoadSigns {...props} ranges={ranges} sourceIndex={117}
            driverPosition={project({ x: 380, y: 200 })} /></svg>);
        const boards = screen.getAllByTestId('comparison-label-sign').map((sign) => {
            const [x, y] = sign.getAttribute('transform')!.match(/-?[\d.]+/g)!.map(Number);
            const rect = within(sign).getByTestId('comparison-label-sign-board');
            return { x, y, width: Number(rect.getAttribute('width')), height: Number(rect.getAttribute('height')) };
        });
        expect(boards).toHaveLength(2);
        expect(boards[0]).toEqual(boards[1]);
        expectAnchor(signsInState('released')[0], 380, 200);
        expectAnchor(signsInState('following')[0], 380, 200);
    });

    it.each([1, 1.6])('lets the merged board leave the viewport without repositioning at scale %s', (scale) => {
        const signAtPosition = (x: number, y: number) => <svg><DriverExpertComparisonRoadSigns {...props}
            sourceIndex={113} driverPosition={project({ x, y })} scale={scale} /></svg>;
        const view = render(signAtPosition(320, 180));
        const sign = screen.getByTestId('comparison-label-sign');
        const [x, y] = sign.getAttribute('transform')!.match(/-?[\d.]+/g)!.map(Number);
        view.rerender(signAtPosition(10, 10));
        const [edgeX, edgeY] = sign.getAttribute('transform')!.match(/-?[\d.]+/g)!.map(Number);
        expect(edgeX).toBeLessThan(0);
        expect(edgeY).toBeLessThan(0);
        expect(edgeX - 10).toBeCloseTo(x - 320);
        expect(edgeY - 10).toBeCloseTo(y - 180);
        expectAnchor(signsInState('following')[0], 10, 10);
    });
});
