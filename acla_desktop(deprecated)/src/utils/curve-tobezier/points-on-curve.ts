export type Point = [number, number];

// distance between 2 points
function distance(p1: Point, p2: Point): number {
  return Math.sqrt(distanceSq(p1, p2));
}

// distance between 2 points squared
function distanceSq(p1: Point, p2: Point): number {
  return Math.pow(p1[0] - p2[0], 2) + Math.pow(p1[1] - p2[1], 2);
}

// Sistance squared from a point p to the line segment vw
function distanceToSegmentSq(p: Point, v: Point, w: Point): number {
  const l2 = distanceSq(v, w);
  if (l2 === 0) {
    return distanceSq(p, v);
  }
  let t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2;
  t = Math.max(0, Math.min(1, t));
  return distanceSq(p, lerp(v, w, t));
}

function lerp(a: Point, b: Point, t: number): Point {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
  ];
}

// Adapted from https://seant23.wordpress.com/2010/11/12/offset-bezier-curves/
function flatness(points: readonly Point[], offset: number): number {
  const p1 = points[offset + 0];
  const p2 = points[offset + 1];
  const p3 = points[offset + 2];
  const p4 = points[offset + 3];

  let ux = 3 * p2[0] - 2 * p1[0] - p4[0]; ux *= ux;
  let uy = 3 * p2[1] - 2 * p1[1] - p4[1]; uy *= uy;
  let vx = 3 * p3[0] - 2 * p4[0] - p1[0]; vx *= vx;
  let vy = 3 * p3[1] - 2 * p4[1] - p1[1]; vy *= vy;

  if (ux < vx) {
    ux = vx;
  }

  if (uy < vy) {
    uy = vy;
  }

  return ux + uy;
}

function getPointsOnBezierCurveWithSplitting(points: readonly Point[], offset: number, tolerance: number, newPoints?: Point[]): Point[] {
  const outPoints = newPoints || [];
  if (flatness(points, offset) < tolerance) {
    const p0 = points[offset + 0];
    if (outPoints.length) {
      const d = distance(outPoints[outPoints.length - 1], p0);
      if (d > 1) {
        outPoints.push(p0);
      }
    } else {
      outPoints.push(p0);
    }
    outPoints.push(points[offset + 3]);
  } else {
    // subdivide
    const t = .5;
    const p1 = points[offset + 0];
    const p2 = points[offset + 1];
    const p3 = points[offset + 2];
    const p4 = points[offset + 3];

    const q1 = lerp(p1, p2, t);
    const q2 = lerp(p2, p3, t);
    const q3 = lerp(p3, p4, t);

    const r1 = lerp(q1, q2, t);
    const r2 = lerp(q2, q3, t);

    const red = lerp(r1, r2, t);

    getPointsOnBezierCurveWithSplitting([p1, q1, r1, red], 0, tolerance, outPoints);
    getPointsOnBezierCurveWithSplitting([red, r2, q3, p4], 0, tolerance, outPoints);
  }
  return outPoints;
}

export function simplify(points: readonly Point[], distance: number): Point[] {
  return simplifyPoints(points, 0, points.length, distance);
}

// Ramer–Douglas–Peucker algorithm
// https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
export function simplifyPoints(points: readonly Point[], start: number, end: number, epsilon: number, newPoints?: Point[]): Point[] {
  const outPoints = newPoints || [];

  // find the most distance point from the endpoints
  const s = points[start];
  const e = points[end - 1];
  let maxDistSq = 0;
  let maxNdx = 1;
  for (let i = start + 1; i < end - 1; ++i) {
    const distSq = distanceToSegmentSq(points[i], s, e);
    if (distSq > maxDistSq) {
      maxDistSq = distSq;
      maxNdx = i;
    }
  }

  // if that point is too far, split
  if (Math.sqrt(maxDistSq) > epsilon) {
    simplifyPoints(points, start, maxNdx + 1, epsilon, outPoints);
    simplifyPoints(points, maxNdx, end, epsilon, outPoints);
  } else {
    if (!outPoints.length) {
      outPoints.push(s);
    }
    outPoints.push(e);
  }

  return outPoints;
}

/**
 * outputs a smooth curve
 * @param points two control points for each user input point must be provided, start and end of user input points only has one control point
 * @param tolerance 
 * @param distance 
 * @returns 
 */
export function ConstructAllPointsOnBezierCurves(points: readonly Point[], tolerance: number = 0.15, distance?: number): Point[] {
  const newPoints: Point[] = [];
  const numSegments = (points.length - 1) / 3;
  for (let i = 0; i < numSegments; i++) {
    const offset = i * 3;
    getPointsOnBezierCurveWithSplitting(points, offset, tolerance, newPoints);
  }
  if (distance && distance > 0) {
    return simplifyPoints(newPoints, 0, newPoints.length, distance);
  }
  return newPoints;
}

export function getBezierTangent(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
  const mt = 1 - t;
  return [
    3 * mt * mt * (p1[0] - p0[0]) +
    6 * mt * t * (p2[0] - p1[0]) +
    3 * t * t * (p3[0] - p2[0]),
    3 * mt * mt * (p1[1] - p0[1]) +
    6 * mt * t * (p2[1] - p1[1]) +
    3 * t * t * (p3[1] - p2[1])
  ];
}
// Which control point (P₀–P₃)
export function getControlPointTangent(P0: Point, P1: Point, P2: Point, P3: Point, index: 0 | 1 | 2 | 3): Point {
  switch (index) {
    case 0: return [3 * (P1[0] - P0[0]), 3 * (P1[1] - P0[1])]; // Tangent at P₀
    case 1: return [3 * (P2[0] - P0[0]), 3 * (P2[1] - P0[1])]; // Tangent at P₁
    case 2: return [3 * (P3[0] - P1[0]), 3 * (P3[1] - P1[1])]; // Tangent at P₂
    case 3: return [3 * (P3[0] - P2[0]), 3 * (P3[1] - P2[1])]; // Tangent at P₃
  }
}

// Get the normal vector (perpendicular to tangent)
export function getNormal(tangent: Point, direction: 'left' | 'right' = 'left'): Point {
  return direction === 'left'
    ? [-tangent[1], tangent[0]]  // Left normal
    : [tangent[1], -tangent[0]]; // Right normal
}

// Normalize a vector to unit length
function normalize(v: Point): Point {
  const length = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
  return length > 0 ? [v[0] / length, v[1] / length] : [0, 0];
}

function createOffsetBezier(P0: Point, P1: Point, P2: Point, P3: Point, offsetDistance: number, direction: 'left' | 'right' = 'left'): [Point, Point, Point, Point] {
  // Calculate normals for each control point
  const normals = [
    normalize(getNormal(getControlPointTangent(P0, P1, P2, P3, 0), direction)), // P₀
    normalize(getNormal(getControlPointTangent(P0, P1, P2, P3, 1), direction)), // P₁
    normalize(getNormal(getControlPointTangent(P0, P1, P2, P3, 2), direction)), // P₂
    normalize(getNormal(getControlPointTangent(P0, P1, P2, P3, 3), direction))  // P₃
  ];

  // Offset each control point
  const Q0: Point = [P0[0] + normals[0][0] * offsetDistance, P0[1] + normals[0][1] * offsetDistance];
  const Q1: Point = [P1[0] + normals[1][0] * offsetDistance, P1[1] + normals[1][1] * offsetDistance];
  const Q2: Point = [P2[0] + normals[2][0] * offsetDistance, P2[1] + normals[2][1] * offsetDistance];
  const Q3: Point = [P3[0] + normals[3][0] * offsetDistance, P3[1] + normals[3][1] * offsetDistance];

  return [Q0, Q1, Q2, Q3];
}

//Array of points (length = 3n + 1, e.g., 4, 7, 10...)
export function offsetBezierPoints(points: Point[], offsetDistance: number, direction: 'left' | 'right' = 'left'): Point[] {
  const offsetPoints: Point[] = [];

  for (let i = 0; i < points.length - 1; i += 3) {
    const P0 = points[i];
    const P1 = points[i + 1];
    const P2 = points[i + 2];
    const P3 = points[i + 3];

    const [Q0, Q1, Q2, Q3] = createOffsetBezier(P0, P1, P2, P3, offsetDistance, direction);

    // Only push Q0 if it's the first segment to avoid duplicates
    if (i === 0) offsetPoints.push(Q0);
    offsetPoints.push(Q1, Q2, Q3);
  }
  return offsetPoints;
}