import { Point } from './points-on-curve.js';

function clone(p: Point): Point {
  return { ...p };
}

/**
 * A Set of points to a set of points representing bezier curves These could be passed to pointsOnBezierCurves function to get the points on the curve
 * @param pointsIn A set of points
 * @param curveTightness Tightness 
 * @returns  A set of points passing through the curve
 */
export function curveToBezier(pointsIn: readonly Point[], curveTightness = 0): Point[] {
  const len = pointsIn.length;
  if (len < 3) {
    throw new Error('A curve must have at least three points.');
  }
  const out: Point[] = [];
  if (len === 3) {
    out.push(
      clone(pointsIn[0]),
      clone(pointsIn[1]),
      clone(pointsIn[2]),
      clone(pointsIn[2])
    );
  } else {
    const points: Point[] = [];
    points.push(pointsIn[0], pointsIn[0]);
    for (let i = 1; i < pointsIn.length; i++) {
      points.push(pointsIn[i]);
      if (i === (pointsIn.length - 1)) {
        points.push(pointsIn[i]);
      }
    }
    const b: Point[] = [];
    const s = 1 - curveTightness;
    out.push(clone(points[0]));
    for (let i = 1; (i + 2) < points.length; i++) {
      const cachedVertArray = points[i];
      b[0] = { x: cachedVertArray.x, y: cachedVertArray.y };
      b[1] = { x: cachedVertArray.x + (s * points[i + 1].x - s * points[i - 1].x) / 6, y: cachedVertArray.y + (s * points[i + 1].y - s * points[i - 1].y) / 6 };
      b[2] = { x: points[i + 1].x + (s * points[i].x - s * points[i + 2].x) / 6, y: points[i + 1].y + (s * points[i].y - s * points[i + 2].y) / 6 };
      b[3] = { x: points[i + 1].x, y: points[i + 1].y };
      out.push(b[1], b[2], b[3]);
    }
  }
  return out;
}