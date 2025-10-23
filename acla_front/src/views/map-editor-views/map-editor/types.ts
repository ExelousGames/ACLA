import { Point } from 'utils/curve-tobezier/points-on-curve';

export type RacingTurningPoint = {
    position: Point;
    type: number;
    index: number;
    description?: string;
    info?: string;
    variables?: { key: string; value: string }[];
};

export type CurbTurningPoint = { id: number; position: Point };
export type BezierPoints = { id: number; position: Point };
export type RacingLinePoint = { id: number; position: Point };
