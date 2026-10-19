/**
 * YOLOv8 + DeepSORT Real-Time Multi-Target C-UAS Tracker
 * 
 * Features:
 * - Vectorized Non-Maximum Suppression (NMS)
 * - IoU Cost Matrix Calculation & Bipartite Greedy Matching
 * - 8-State Extended Kalman Bounding Box Filter [xc, yc, a, h, vxc, vyc, va, vh]
 * - Appearance Embedding Metric (Cosine Distance) for Re-Identification
 * - C-UAS Target Classification (Rotary UAV, Fixed-Wing Loitering Munition, Micro-Drone, Decoy)
 */

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
  className: string;
}

export type TrackState = 'TENTATIVE' | 'CONFIRMED' | 'DELETED';

export interface TrackedTarget {
  trackId: number;
  bbox: BoundingBox;
  state: TrackState;
  age: number;
  hits: number;
  timeSinceUpdate: number;
  velocity: { vx: number; vy: number };
  featureVector?: number[];
  classification: string;
}

/**
 * Compute Intersection over Union (IoU) of two bounding boxes.
 */
export function computeIoU(boxA: BoundingBox, boxB: BoundingBox): number {
  const xA = Math.max(boxA.x1, boxB.x1);
  const yA = Math.max(boxA.y1, boxB.y1);
  const xB = Math.min(boxA.x2, boxB.x2);
  const yB = Math.min(boxA.y2, boxB.y2);

  const interWidth = Math.max(0, xB - xA);
  const interHeight = Math.max(0, yB - yA);
  const interArea = interWidth * interHeight;

  const boxAArea = Math.max(0, boxA.x2 - boxA.x1) * Math.max(0, boxA.y2 - boxA.y1);
  const boxBArea = Math.max(0, boxB.x2 - boxB.x1) * Math.max(0, boxB.y2 - boxB.y1);
  const unionArea = boxAArea + boxBArea - interArea;

  if (unionArea <= 0) return 0;
  return interArea / unionArea;
}

/**
 * Non-Maximum Suppression (NMS) to eliminate duplicate detections.
 */
export function nonMaximumSuppression(
  boxes: BoundingBox[],
  iouThreshold = 0.45,
  scoreThreshold = 0.25
): BoundingBox[] {
  const filtered = boxes.filter(b => b.score >= scoreThreshold);
  filtered.sort((a, b) => b.score - a.score);

  const picked: BoundingBox[] = [];
  const suppressed = new Array<boolean>(filtered.length).fill(false);

  for (let i = 0; i < filtered.length; i++) {
    if (suppressed[i]) continue;
    const current = filtered[i];
    picked.push(current);

    for (let j = i + 1; j < filtered.length; j++) {
      if (suppressed[j]) continue;
      // Per-class suppression (or cross-class if needed)
      if (current.classId === filtered[j].classId) {
        const iou = computeIoU(current, filtered[j]);
        if (iou >= iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
  }

  return picked;
}

/**
 * Computes pairwise IoU distance cost matrix: cost = 1.0 - IoU
 */
export function computeIoUCostMatrix(
  trackBoxes: BoundingBox[],
  detBoxes: BoundingBox[]
): number[][] {
  const matrix: number[][] = [];
  for (let i = 0; i < trackBoxes.length; i++) {
    matrix[i] = [];
    for (let j = 0; j < detBoxes.length; j++) {
      matrix[i][j] = 1.0 - computeIoU(trackBoxes[i], detBoxes[j]);
    }
  }
  return matrix;
}

/**
 * Computes Cosine Distance between two L2-normalized feature vectors.
 */
export function computeCosineDistance(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length || vecA.length === 0) return 1.0;
  let dot = 0.0;
  let normA = 0.0;
  let normB = 0.0;

  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) return 1.0;
  const similarity = dot / (normA * normB);
  return Math.max(0, 1.0 - similarity);
}

/**
 * Bipartite Minimum-Cost Greedy Association
 */
export function greedyMinCostAssignment(
  costMatrix: number[][],
  maxThreshold: number
): { matches: [number, number][]; unmatchedA: number[]; unmatchedB: number[] } {
  const numA = costMatrix.length;
  if (numA === 0) {
    return { matches: [], unmatchedA: [], unmatchedB: [] };
  }
  const numB = costMatrix[0].length;
  if (numB === 0) {
    return { matches: [], unmatchedA: Array.from({ length: numA }, (_, i) => i), unmatchedB: [] };
  }

  // Flatten costs with indices
  const entries: { r: number; c: number; cost: number }[] = [];
  for (let r = 0; r < numA; r++) {
    for (let c = 0; c < numB; c++) {
      entries.push({ r, c, cost: costMatrix[r][c] });
    }
  }
  entries.sort((a, b) => a.cost - b.cost);

  const matchedA = new Set<number>();
  const matchedB = new Set<number>();
  const matches: [number, number][] = [];

  for (const { r, c, cost } of entries) {
    if (cost > maxThreshold) break;
    if (!matchedA.has(r) && !matchedB.has(c)) {
      matchedA.add(r);
      matchedB.add(c);
      matches.push([r, c]);
    }
  }

  const unmatchedA: number[] = [];
  for (let i = 0; i < numA; i++) {
    if (!matchedA.has(i)) unmatchedA.push(i);
  }

  const unmatchedB: number[] = [];
  for (let j = 0; j < numB; j++) {
    if (!matchedB.has(j)) unmatchedB.push(j);
  }

  return { matches, unmatchedA, unmatchedB };
}

/**
 * 8-State Kalman Filter for Bounding Box Tracking:
 * State x: [xc, yc, a, h, vxc, vyc, va, vh]^T
 * xc, yc: center coordinates
 * a: aspect ratio (w / h)
 * h: height
 * v*: corresponding velocities
 */
export class KalmanBoxTracker {
  private static count = 0;
  public id: number;
  public state: number[]; // 8 elements
  public covariance: number[][]; // 8x8 matrix
  public timeSinceUpdate = 0;
  public hits = 0;
  public age = 0;
  public trackState: TrackState = 'TENTATIVE';
  public featureVector?: number[];
  public classId: number;
  public className: string;
  public score: number;

  constructor(initialBox: BoundingBox, feature?: number[]) {
    this.id = ++KalmanBoxTracker.count;
    this.classId = initialBox.classId;
    this.className = initialBox.className;
    this.score = initialBox.score;
    this.featureVector = feature;

    const w = Math.max(1, initialBox.x2 - initialBox.x1);
    const h = Math.max(1, initialBox.y2 - initialBox.y1);
    const xc = initialBox.x1 + w / 2;
    const yc = initialBox.y1 + h / 2;
    const a = w / h;

    this.state = [xc, yc, a, h, 0, 0, 0, 0];

    // Initialize 8x8 covariance matrix P
    this.covariance = Array.from({ length: 8 }, (_, i) =>
      Array.from({ length: 8 }, (_, j) => {
        if (i !== j) return 0;
        return i < 4 ? 10.0 : 100.0;
      })
    );
  }

  /**
   * Predict the next bounding box state with constant velocity model
   */
  public predict(dt = 1.0): BoundingBox {
    // Constant velocity update: pos = pos + vel * dt
    for (let i = 0; i < 4; i++) {
      this.state[i] += this.state[i + 4] * dt;
    }

    this.age += 1;
    this.timeSinceUpdate += 1;

    // Add process noise Q to covariance diagonal
    const qPos = 1.0;
    const qVel = 4.0;
    for (let i = 0; i < 8; i++) {
      this.covariance[i][i] += i < 4 ? qPos : qVel;
    }

    return this.toBoundingBox();
  }

  /**
   * Update tracker state with measurement [xc, yc, a, h]
   */
  public update(box: BoundingBox, feature?: number[]): void {
    const w = Math.max(1, box.x2 - box.x1);
    const h = Math.max(1, box.y2 - box.y1);
    const xc = box.x1 + w / 2;
    const yc = box.y1 + h / 2;
    const a = w / h;

    const z = [xc, yc, a, h];

    // Innovation (residual) y = z - H*x
    const y: number[] = [];
    for (let i = 0; i < 4; i++) {
      y[i] = z[i] - this.state[i];
    }

    // Measurement noise R
    const rNoise = [2.0, 2.0, 0.05, 2.0];

    // Simplified Kalman gain K_i = P_ii / (P_ii + R_ii) for position, with velocity coupling
    for (let i = 0; i < 4; i++) {
      const denom = this.covariance[i][i] + rNoise[i];
      const kPos = this.covariance[i][i] / denom;
      const kVel = 0.5 * kPos;

      this.state[i] += kPos * y[i];
      this.state[i + 4] += kVel * y[i];

      this.covariance[i][i] *= (1 - kPos);
      this.covariance[i + 4][i + 4] *= (1 - kVel * 0.5);
    }

    // Keep aspect ratio & height positive
    if (this.state[2] <= 0) this.state[2] = 0.1;
    if (this.state[3] <= 0) this.state[3] = 1.0;

    this.timeSinceUpdate = 0;
    this.hits += 1;
    this.score = box.score;
    this.classId = box.classId;
    this.className = box.className;

    // Exponential moving average for appearance feature vector
    if (feature && feature.length > 0) {
      if (!this.featureVector) {
        this.featureVector = [...feature];
      } else {
        const alpha = 0.8;
        for (let i = 0; i < feature.length; i++) {
          this.featureVector[i] = alpha * this.featureVector[i] + (1 - alpha) * feature[i];
        }
      }
    }
  }

  /**
   * Convert current state to bounding box
   */
  public toBoundingBox(): BoundingBox {
    const [xc, yc, a, h] = this.state;
    const w = a * h;
    return {
      x1: xc - w / 2,
      y1: yc - h / 2,
      x2: xc + w / 2,
      y2: yc + h / 2,
      score: this.score,
      classId: this.classId,
      className: this.className,
    };
  }
}

/**
 * Complete DeepSORT Multi-Object Tracking Pipeline
 */
export class DeepSortTracker {
  private trackers: KalmanBoxTracker[] = [];
  private maxAge: number;
  private minHits: number;
  private iouThreshold: number;
  private appearanceThreshold: number;

  constructor(options?: {
    maxAge?: number;
    minHits?: number;
    iouThreshold?: number;
    appearanceThreshold?: number;
  }) {
    this.maxAge = options?.maxAge ?? 30;
    this.minHits = options?.minHits ?? 3;
    this.iouThreshold = options?.iouThreshold ?? 0.7;
    this.appearanceThreshold = options?.appearanceThreshold ?? 0.35;
  }

  /**
   * Execute tracking iteration on new detections
   */
  public update(detections: BoundingBox[], features?: number[][]): TrackedTarget[] {
    // 1. Predict all current tracks
    for (const tracker of this.trackers) {
      tracker.predict();
    }

    // 2. Cascade matching:
    // Try matching using appearance features first if available
    let unmatchedTrackIndices = Array.from({ length: this.trackers.length }, (_, i) => i);
    let unmatchedDetIndices = Array.from({ length: detections.length }, (_, j) => j);
    const confirmedMatches: [number, number][] = [];

    if (features && features.length === detections.length && this.trackers.length > 0) {
      const activeTrackers = this.trackers.filter(t => t.featureVector != null);
      if (activeTrackers.length > 0) {
        const costMat: number[][] = [];
        for (let i = 0; i < this.trackers.length; i++) {
          costMat[i] = [];
          for (let j = 0; j < detections.length; j++) {
            if (this.trackers[i].featureVector && features[j]) {
              costMat[i][j] = computeCosineDistance(this.trackers[i].featureVector!, features[j]);
            } else {
              costMat[i][j] = 1.0;
            }
          }
        }

        const appMatches = greedyMinCostAssignment(costMat, this.appearanceThreshold);
        for (const [tIdx, dIdx] of appMatches.matches) {
          confirmedMatches.push([tIdx, dIdx]);
        }
        unmatchedTrackIndices = appMatches.unmatchedA;
        unmatchedDetIndices = appMatches.unmatchedB;
      }
    }

    // 3. Fallback matching via IoU for remaining tracks & detections
    if (unmatchedTrackIndices.length > 0 && unmatchedDetIndices.length > 0) {
      const subTrackBoxes = unmatchedTrackIndices.map(i => this.trackers[i].toBoundingBox());
      const subDetBoxes = unmatchedDetIndices.map(j => detections[j]);

      const iouCostMat = computeIoUCostMatrix(subTrackBoxes, subDetBoxes);
      const iouMatches = greedyMinCostAssignment(iouCostMat, this.iouThreshold);

      for (const [subT, subD] of iouMatches.matches) {
        confirmedMatches.push([unmatchedTrackIndices[subT], unmatchedDetIndices[subD]]);
      }

      const newlyMatchedT = new Set(iouMatches.matches.map(m => m[0]));
      const newlyMatchedD = new Set(iouMatches.matches.map(m => m[1]));

      unmatchedTrackIndices = unmatchedTrackIndices.filter((_, idx) => !newlyMatchedT.has(idx));
      unmatchedDetIndices = unmatchedDetIndices.filter((_, idx) => !newlyMatchedD.has(idx));
    }

    // 4. Update matched tracks
    for (const [tIdx, dIdx] of confirmedMatches) {
      const det = detections[dIdx];
      const feat = features ? features[dIdx] : undefined;
      this.trackers[tIdx].update(det, feat);
      if (this.trackers[tIdx].hits >= this.minHits) {
        this.trackers[tIdx].trackState = 'CONFIRMED';
      }
    }

    // 5. Initiate new tracks from unmatched detections
    for (const dIdx of unmatchedDetIndices) {
      const det = detections[dIdx];
      const feat = features ? features[dIdx] : undefined;
      const newTracker = new KalmanBoxTracker(det, feat);
      this.trackers.push(newTracker);
    }

    // 6. Delete dead tracks
    this.trackers = this.trackers.filter(t => {
      if (t.timeSinceUpdate > this.maxAge) {
        t.trackState = 'DELETED';
        return false;
      }
      return true;
    });

    // 7. Format active target output
    return this.trackers
      .filter(t => t.trackState === 'CONFIRMED' || t.hits >= 2)
      .map(t => {
        const box = t.toBoundingBox();
        return {
          trackId: t.id,
          bbox: box,
          state: t.trackState,
          age: t.age,
          hits: t.hits,
          timeSinceUpdate: t.timeSinceUpdate,
          velocity: {
            vx: t.state[4],
            vy: t.state[5],
          },
          featureVector: t.featureVector,
          classification: mapClassIdToCUASLabel(t.classId, t.className),
        };
      });
  }

  public getActiveTracksCount(): number {
    return this.trackers.length;
  }
}

/**
 * Maps standard vision class labels to military C-UAS threat categories
 */
export function mapClassIdToCUASLabel(classId: number, rawName?: string): string {
  const norm = (rawName || '').toLowerCase();
  if (norm.includes('quad') || norm.includes('rotary') || classId === 1) {
    return 'UAV_ROTARY_QUAD';
  }
  if (norm.includes('fixed') || norm.includes('shahed') || norm.includes('loitering') || classId === 2) {
    return 'LOITERING_MUNITION_FIXED_WING';
  }
  if (norm.includes('fpv') || norm.includes('micro') || classId === 3) {
    return 'UAV_FPV_STRIKE';
  }
  if (norm.includes('bird') || norm.includes('avian') || classId === 4) {
    return 'BIOLOGICAL_DECOY_BIRD';
  }
  if (norm.includes('vehicle') || norm.includes('car') || classId === 5) {
    return 'GROUND_VEHICLE_INTRUDER';
  }
  if (norm.includes('person') || norm.includes('human') || classId === 6) {
    return 'PERSONNEL_PERIMETER_BREACH';
  }
  return 'UAS_UNKNOWN_AIRBORNE';
}
