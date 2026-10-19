import { describe, it, expect } from 'vitest';
import {
  computeIoU,
  nonMaximumSuppression,
  KalmanBoxTracker,
  DeepSortTracker,
  BoundingBox,
} from '../src/vision/yolov8_deepsort_tracker';
import {
  calculateSpeedOfSound,
  computeGCCPHAT,
  AcousticMUSICArray,
  createCircularMicArray,
} from '../src/cuas/acoustic_doa_array';
import {
  HighEnergyLaserEffector,
  RFJammerEffector,
  ProportionalNavigationGuidance,
  Vec3,
} from '../src/cuas/kinetic_ew_effector';
import {
  isPointInPolygon2D,
  PerimeterGeofenceEngine,
  LiDARClusteringDBSCAN,
  LiDARPoint,
} from '../src/geofence/perimeter_zone_engine';

describe('YOLOv8 + DeepSORT Tracker Unit Suite', () => {
  it('computes exact IoU for overlapping and disjoint bounding boxes', () => {
    const boxA: BoundingBox = { x1: 0, y1: 0, x2: 100, y2: 100, score: 0.9, classId: 1, className: 'drone' };
    const boxB: BoundingBox = { x1: 0, y1: 0, x2: 100, y2: 100, score: 0.85, classId: 1, className: 'drone' };
    expect(computeIoU(boxA, boxB)).toBeCloseTo(1.0, 5);

    const boxDisjoint: BoundingBox = { x1: 200, y1: 200, x2: 300, y2: 300, score: 0.8, classId: 1, className: 'drone' };
    expect(computeIoU(boxA, boxDisjoint)).toBe(0.0);

    const boxHalf: BoundingBox = { x1: 0, y1: 0, x2: 50, y2: 100, score: 0.8, classId: 1, className: 'drone' };
    // boxA area: 10000, boxHalf area: 5000, inter: 5000, union: 10000 -> IoU = 0.5
    expect(computeIoU(boxA, boxHalf)).toBeCloseTo(0.5, 4);
  });

  it('performs Non-Maximum Suppression (NMS) suppressing duplicate lower-score detections', () => {
    const candidates: BoundingBox[] = [
      { x1: 10, y1: 10, x2: 60, y2: 60, score: 0.95, classId: 1, className: 'uav' },
      { x1: 12, y1: 11, x2: 62, y2: 59, score: 0.82, classId: 1, className: 'uav' }, // duplicate
      { x1: 200, y1: 200, x2: 250, y2: 250, score: 0.88, classId: 1, className: 'uav' }, // distinct target
    ];

    const result = nonMaximumSuppression(candidates, 0.45, 0.25);
    expect(result.length).toBe(2);
    expect(result[0].score).toBe(0.95);
    expect(result[1].score).toBe(0.88);
  });

  it('initializes and updates KalmanBoxTracker state vector', () => {
    const initBox: BoundingBox = { x1: 100, y1: 100, x2: 140, y2: 140, score: 0.9, classId: 1, className: 'uav' };
    const tracker = new KalmanBoxTracker(initBox);

    const predicted = tracker.predict();
    expect(predicted.x1).toBeCloseTo(initBox.x1, 1);

    const measurement: BoundingBox = { x1: 105, y1: 105, x2: 145, y2: 145, score: 0.92, classId: 1, className: 'uav' };
    tracker.update(measurement);
    expect(tracker.hits).toBe(1);
    expect(tracker.timeSinceUpdate).toBe(0);
  });

  it('tracks multi-frame UAS trajectory using DeepSortTracker', () => {
    const tracker = new DeepSortTracker({ minHits: 2, maxAge: 5 });

    const frame1: BoundingBox[] = [
      { x1: 100, y1: 100, x2: 140, y2: 140, score: 0.9, classId: 1, className: 'UAV_ROTARY' },
    ];
    tracker.update(frame1);

    const frame2: BoundingBox[] = [
      { x1: 104, y1: 102, x2: 144, y2: 142, score: 0.92, classId: 1, className: 'UAV_ROTARY' },
    ];
    const confirmedTracks = tracker.update(frame2);

    expect(confirmedTracks.length).toBe(1);
    expect(confirmedTracks[0].hits).toBe(2);
    expect(confirmedTracks[0].classification).toBe('UAV_ROTARY_QUAD');
  });
});

describe('Acoustic Phased Array & DoA Suite', () => {
  it('calculates speed of sound according to temperature', () => {
    const c20 = calculateSpeedOfSound(20);
    expect(c20).toBeGreaterThan(340);
    expect(c20).toBeLessThan(345);

    const c0 = calculateSpeedOfSound(0);
    expect(c0).toBeCloseTo(331.3, 1);
  });

  it('estimates TDoA using GCC-PHAT on correlated synthetic waveforms', () => {
    const N = 128;
    const sigA = Array.from({ length: N }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / 8000));
    // Zero delay
    const resultZero = computeGCCPHAT(sigA, sigA, 8000);
    expect(resultZero.tdoaSeconds).toBe(0);
  });

  it('computes MUSIC spatial array steering vector and executes DOA estimation', () => {
    const micArray = createCircularMicArray(8, 0.25);
    const musicSolver = new AcousticMUSICArray(micArray, 20);

    const testFreq = 300; // Hz (Drone motor harmonic)
    const snapshots = Array.from({ length: 8 }, () =>
      musicSolver.synthesizeSourceSnapshot(90, 30, testFreq, 30)
    );

    const result = musicSolver.estimateDOA(snapshots, testFreq);
    expect(result.dominantFrequencyHz).toBe(testFreq);
    expect(result.targetType).toBe('DRONE_ROTOR');
    expect(result.azimuthDeg).toBeGreaterThanOrEqual(0);
    expect(result.azimuthDeg).toBeLessThanOrEqual(360);
  });
});

describe('Kinetic & Directed Energy Effector Suite', () => {
  it('computes High Energy Laser dwell time, irradiance and lethal envelope', () => {
    const laser = new HighEnergyLaserEffector({ powerWatts: 50000 });
    const resultNear = laser.calculateDwellTime(800, 'CFRP_COMPOSITE');

    expect(resultNear.powerOnTargetWatts).toBeGreaterThan(40000);
    expect(resultNear.requiredDwellTimeSec).toBeLessThan(2.0);
    expect(resultNear.isLethalAtRange).toBe(true);

    const resultFar = laser.calculateDwellTime(3500, 'CFRP_COMPOSITE');
    expect(resultFar.isLethalAtRange).toBe(false);
  });

  it('computes RF Jammer Jam-to-Signal ratio and link denial threshold', () => {
    const jammer = new RFJammerEffector();
    // Drone at 600m from jammer, 1800m from enemy operator
    const budget = jammer.computeLinkBudget('ISM_2_4GHZ', 600, 1800);

    expect(budget.jamToSignalRatioDb).toBeGreaterThan(15.0);
    expect(budget.isLinkDenied).toBe(true);
    expect(budget.recommendedAction).toContain('C2 LINK SEVERED');
  });

  it('solves 3D Proportional Navigation guidance acceleration toward maneuvering threat', () => {
    const pnSolver = new ProportionalNavigationGuidance(4.0, 15.0);

    const interceptorPos = { x: 0, y: 0, z: 50 };
    const interceptorVel = { x: 120, y: 0, z: 0 }; // 120 m/s sprint

    const targetPos = { x: 800, y: 150, z: 120 };
    const targetVel = { x: -40, y: -20, z: 0 }; // Inbound hostile drone

    const cmd = pnSolver.computeGuidance(interceptorPos, interceptorVel, targetPos, targetVel);

    expect(cmd.closingVelocity).toBeGreaterThan(100);
    expect(cmd.accelMagnitudeG).toBeGreaterThan(0);
    expect(cmd.accelMagnitudeG).toBeLessThanOrEqual(15.0);
    expect(cmd.timeToGoSec).toBeGreaterThan(0);
  });
});

describe('Perimeter Geofence & LiDAR Clustering Suite', () => {
  it('verifies point-in-polygon ray-casting algorithm for boundary checks', () => {
    const square = [
      { x: -50, y: -50 },
      { x: 50, y: -50 },
      { x: 50, y: 50 },
      { x: -50, y: 50 },
    ];

    expect(isPointInPolygon2D({ x: 0, y: 0 }, square)).toBe(true);
    expect(isPointInPolygon2D({ x: 25, y: -25 }, square)).toBe(true);
    expect(isPointInPolygon2D({ x: 100, y: 0 }, square)).toBe(false);
  });

  it('evaluates multi-tier military ROE zones for inbound targets', () => {
    const geofence = new PerimeterGeofenceEngine();

    // Target inside command center sanctum (Tier 0)
    const sanctumEval = geofence.evaluateTrack({ x: 50, y: 50, z: 30 });
    expect(sanctumEval.isInside).toBe(true);
    expect(sanctumEval.tier).toBe('TIER_0_HVA_SANCTUM');
    expect(sanctumEval.threatLevel).toBe('CRITICAL_LETHAL');

    // Target outside all zones heading inbound
    const distantEval = geofence.evaluateTrack(
      { x: 3500, y: 0, z: 200 },
      { x: -50, y: 0, z: 0 } // Inbound speed 50 m/s
    );
    expect(distantEval.isInside).toBe(false);
    expect(distantEval.timeToBreachSec).toBeDefined();
    expect(distantEval.timeToBreachSec!).toBeGreaterThan(0);
  });

  it('clusters 3D LiDAR point cloud using DBSCAN and extracts target bounding box', () => {
    const dbscan = new LiDARClusteringDBSCAN(1.5, 3);

    // Dense cluster of 5 points representing a micro drone at (100, 50, 20)
    const dronePoints: LiDARPoint[] = [
      { x: 100.0, y: 50.0, z: 20.0 },
      { x: 100.2, y: 50.1, z: 20.1 },
      { x: 99.8, y: 49.9, z: 19.9 },
      { x: 100.1, y: 50.3, z: 20.2 },
      { x: 99.9, y: 50.0, z: 20.0 },
    ];

    // Outlier noise point
    const noisePoint: LiDARPoint = { x: 500.0, y: 500.0, z: 100.0 };

    const clusters = dbscan.cluster([...dronePoints, noisePoint]);
    expect(clusters.length).toBe(1);
    expect(clusters[0].centroid.x).toBeCloseTo(100.0, 1);
    expect(clusters[0].centroid.z).toBeCloseTo(20.0, 1);
    expect(clusters[0].classification).toBe('MICRO_DRONE');
  });
});
