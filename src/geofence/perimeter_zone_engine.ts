/**
 * Multi-Tier Perimeter Geofence & 3D LiDAR DBSCAN Clustering Engine
 * 
 * Features:
 * - Multi-Tier Rules of Engagement (ROE) Polygon Geofencing
 * - Ray-Casting Point-in-Polygon & Altitude Envelope Validation
 * - Trajectory Vector Boundary Penetration Prediction (Time-to-Breach)
 * - 3D LiDAR Point Cloud DBSCAN Spatial Clustering & Bounding Box Extraction
 */

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export type ROETier = 'TIER_1_SURVEILLANCE' | 'TIER_2_EXCLUSION' | 'TIER_3_LETHAL_KILL' | 'TIER_0_HVA_SANCTUM';

export interface GeofenceZone {
  id: string;
  name: string;
  tier: ROETier;
  polygon: Point2D[];
  minAltitudeMeters: number;
  maxAltitudeMeters: number;
  roeProtocol: string;
  colorHex: string;
}

export interface ZoneEvaluationResult {
  isInside: boolean;
  activeZoneId?: string;
  tier?: ROETier;
  distanceToBoundaryMeters: number;
  timeToBreachSec?: number;
  threatLevel: 'NOMINAL' | 'ADVISORY' | 'WARNING' | 'CRITICAL_LETHAL';
  recommendedIntervention: string;
}

/**
 * Standard Ray-Casting algorithm to check if 2D point is inside a polygon
 */
export function isPointInPolygon2D(point: Point2D, polygon: Point2D[]): boolean {
  let inside = false;
  const n = polygon.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    const intersect = yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi || 1e-9) + xi;
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Computes minimum Euclidean distance from a 2D point to any segment of a polygon
 */
export function distanceToPolygonBoundary2D(point: Point2D, polygon: Point2D[]): number {
  let minDistance = Infinity;
  const n = polygon.length;

  for (let i = 0; i < n; i++) {
    const p1 = polygon[i];
    const p2 = polygon[(i + 1) % n];

    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const segLenSq = dx * dx + dy * dy;

    let t = segLenSq === 0 ? 0 : ((point.x - p1.x) * dx + (point.y - p1.y) * dy) / segLenSq;
    t = Math.max(0, Math.min(1, t));

    const projX = p1.x + t * dx;
    const projY = p1.y + t * dy;
    const dist = Math.hypot(point.x - projX, point.y - projY);

    if (dist < minDistance) {
      minDistance = dist;
    }
  }

  return minDistance;
}

/**
 * Perimeter Geofence Engine with Multi-Tier Tactical ROE Evaluation
 */
export class PerimeterGeofenceEngine {
  private zones: GeofenceZone[] = [];

  constructor(customZones?: GeofenceZone[]) {
    this.zones = customZones ?? this.getDefaultTacticalZones();
  }

  /**
   * Evaluates an airborne or ground track against all base perimeter geofences
   */
  public evaluateTrack(
    position: Point3D,
    velocity: Point3D = { x: 0, y: 0, z: 0 }
  ): ZoneEvaluationResult {
    const pt2D: Point2D = { x: position.x, y: position.y };

    // Search from most restrictive (Tier 0 / Tier 3) to least restrictive (Tier 1)
    const tierPriority: Record<ROETier, number> = {
      TIER_0_HVA_SANCTUM: 4,
      TIER_3_LETHAL_KILL: 3,
      TIER_2_EXCLUSION: 2,
      TIER_1_SURVEILLANCE: 1,
    };

    let matchedZone: GeofenceZone | null = null;
    let highestPriority = -1;

    for (const zone of this.zones) {
      if (position.z >= zone.minAltitudeMeters && position.z <= zone.maxAltitudeMeters) {
        if (isPointInPolygon2D(pt2D, zone.polygon)) {
          const prio = tierPriority[zone.tier];
          if (prio > highestPriority) {
            highestPriority = prio;
            matchedZone = zone;
          }
        }
      }
    }

    if (matchedZone) {
      const distToBoundary = distanceToPolygonBoundary2D(pt2D, matchedZone.polygon);
      let threatLevel: ZoneEvaluationResult['threatLevel'] = 'ADVISORY';
      let recommendedIntervention = matchedZone.roeProtocol;

      if (matchedZone.tier === 'TIER_0_HVA_SANCTUM' || matchedZone.tier === 'TIER_3_LETHAL_KILL') {
        threatLevel = 'CRITICAL_LETHAL';
      } else if (matchedZone.tier === 'TIER_2_EXCLUSION') {
        threatLevel = 'WARNING';
      }

      return {
        isInside: true,
        activeZoneId: matchedZone.id,
        tier: matchedZone.tier,
        distanceToBoundaryMeters: Math.round(distToBoundary * 10) / 10,
        threatLevel,
        recommendedIntervention,
      };
    }

    // Outside all zones: compute distance to Tier 1 outer perimeter and Time-To-Breach
    const outerZone = this.zones.find(z => z.tier === 'TIER_1_SURVEILLANCE') || this.zones[0];
    const distToOuter = distanceToPolygonBoundary2D(pt2D, outerZone.polygon);

    let timeToBreachSec: number | undefined;
    const speed = Math.hypot(velocity.x, velocity.y);
    if (speed > 1.0) {
      // Check if velocity is heading towards the perimeter
      const projectedX = pt2D.x + velocity.x;
      const projectedY = pt2D.y + velocity.y;
      const distProjected = distanceToPolygonBoundary2D({ x: projectedX, y: projectedY }, outerZone.polygon);
      if (distProjected < distToOuter) {
        timeToBreachSec = Math.round((distToOuter / speed) * 10) / 10;
      }
    }

    return {
      isInside: false,
      distanceToBoundaryMeters: Math.round(distToOuter * 10) / 10,
      timeToBreachSec,
      threatLevel: timeToBreachSec && timeToBreachSec < 30 ? 'WARNING' : 'NOMINAL',
      recommendedIntervention: 'RADAR TRACKING & EO/IR SLEW-TO-CUE',
    };
  }

  /**
   * Generates default military concentric perimeter zones around coordinate (0,0)
   */
  public getDefaultTacticalZones(): GeofenceZone[] {
    const makeRegularPolygon = (radius: number, numPoints = 16): Point2D[] => {
      const poly: Point2D[] = [];
      for (let i = 0; i < numPoints; i++) {
        const theta = (2 * Math.PI * i) / numPoints;
        poly.push({
          x: radius * Math.cos(theta),
          y: radius * Math.sin(theta),
        });
      }
      return poly;
    };

    return [
      {
        id: 'ZONE_TIER_0',
        name: 'Command Center & Munitions HVA Sanctum',
        tier: 'TIER_0_HVA_SANCTUM',
        polygon: makeRegularPolygon(180, 12),
        minAltitudeMeters: 0,
        maxAltitudeMeters: 500,
        roeProtocol: 'ROE RED: IMMEDIATE LETHAL ENGAGEMENT (CIWS / DEW)',
        colorHex: '#ef4444',
      },
      {
        id: 'ZONE_TIER_3',
        name: 'Inner Base Lethal Defense Zone',
        tier: 'TIER_3_LETHAL_KILL',
        polygon: makeRegularPolygon(500, 16),
        minAltitudeMeters: 0,
        maxAltitudeMeters: 1000,
        roeProtocol: 'ROE AMBER-RED: AUTHORIZED KINETIC INTERCEPT & HIGH POWER LASER',
        colorHex: '#f97316',
      },
      {
        id: 'ZONE_TIER_2',
        name: 'Intermediate Exclusion & RF Jamming Zone',
        tier: 'TIER_2_EXCLUSION',
        polygon: makeRegularPolygon(1200, 20),
        minAltitudeMeters: 0,
        maxAltitudeMeters: 1500,
        roeProtocol: 'ROE YELLOW: ENGAGE DIRECTIONAL EW JAMMERS & PREPARE DEW TRACKING',
        colorHex: '#eab308',
      },
      {
        id: 'ZONE_TIER_1',
        name: 'Outer Early Warning & Surveillance Perimeter',
        tier: 'TIER_1_SURVEILLANCE',
        polygon: makeRegularPolygon(2500, 24),
        minAltitudeMeters: 0,
        maxAltitudeMeters: 2500,
        roeProtocol: 'ROE GREEN: PASSIVE FMCW RADAR & ACOUSTIC ARRAY CONTINUOUS TRACKING',
        colorHex: '#3b82f6',
      },
    ];
  }

  public getZones(): GeofenceZone[] {
    return this.zones;
  }
}

// ==========================================================
// 3D LiDAR DBSCAN Point Cloud Spatial Clustering
// ==========================================================

export interface LiDARPoint extends Point3D {
  intensity?: number;
  clusterId?: number; // -1 for noise
}

export interface LiDARCluster {
  clusterId: number;
  centroid: Point3D;
  boundingBox: {
    min: Point3D;
    max: Point3D;
    dimensions: Point3D; // width, length, height
  };
  numPoints: number;
  classification: 'MICRO_DRONE' | 'MEDIUM_UAV' | 'PERSONNEL' | 'VEHICLE' | 'CLUTTER';
}

export class LiDARClusteringDBSCAN {
  private epsMeters: number;
  private minPoints: number;

  constructor(epsMeters = 2.0, minPoints = 5) {
    this.epsMeters = epsMeters;
    this.minPoints = minPoints;
  }

  /**
   * Executes 3D DBSCAN spatial clustering on LiDAR point coordinates
   */
  public cluster(points: LiDARPoint[]): LiDARCluster[] {
    const N = points.length;
    const visited = new Uint8Array(N);
    const clusterIds = new Int32Array(N).fill(-1); // -1 = unassigned / noise
    let clusterCount = 0;

    const distSq = (a: Point3D, b: Point3D) => {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      return dx * dx + dy * dy + dz * dz;
    };

    const epsSq = this.epsMeters * this.epsMeters;

    const regionQuery = (pointIdx: number): number[] => {
      const neighbors: number[] = [];
      const pt = points[pointIdx];
      for (let i = 0; i < N; i++) {
        if (distSq(pt, points[i]) <= epsSq) {
          neighbors.push(i);
        }
      }
      return neighbors;
    };

    for (let i = 0; i < N; i++) {
      if (visited[i]) continue;
      visited[i] = 1;

      const neighbors = regionQuery(i);
      if (neighbors.length < this.minPoints) {
        clusterIds[i] = -1; // Noise
      } else {
        const cId = clusterCount++;
        clusterIds[i] = cId;

        // Expand cluster
        const queue = [...neighbors];
        let qHead = 0;

        while (qHead < queue.length) {
          const ptIdx = queue[qHead++];
          if (!visited[ptIdx]) {
            visited[ptIdx] = 1;
            const subNeighbors = regionQuery(ptIdx);
            if (subNeighbors.length >= this.minPoints) {
              for (const sn of subNeighbors) {
                if (!visited[sn]) {
                  queue.push(sn);
                }
              }
            }
          }
          if (clusterIds[ptIdx] === -1) {
            clusterIds[ptIdx] = cId;
          }
        }
      }
    }

    // Aggregate points into structured clusters
    const clusterMap = new Map<number, LiDARPoint[]>();
    for (let i = 0; i < N; i++) {
      const cId = clusterIds[i];
      if (cId >= 0) {
        if (!clusterMap.has(cId)) clusterMap.set(cId, []);
        clusterMap.get(cId)!.push(points[i]);
      }
    }

    const results: LiDARCluster[] = [];
    for (const [cId, cPoints] of clusterMap.entries()) {
      let sumX = 0, sumY = 0, sumZ = 0;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

      for (const p of cPoints) {
        sumX += p.x;
        sumY += p.y;
        sumZ += p.z;
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.z < minZ) minZ = p.z;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
        if (p.z > maxZ) maxZ = p.z;
      }

      const count = cPoints.length;
      const centroid: Point3D = {
        x: sumX / count,
        y: sumY / count,
        z: sumZ / count,
      };

      const dimensions: Point3D = {
        x: Math.max(0.1, maxX - minX),
        y: Math.max(0.1, maxY - minY),
        z: Math.max(0.1, maxZ - minZ),
      };

      // Classification rule based on dimensions & altitude
      let classification: LiDARCluster['classification'] = 'CLUTTER';
      const maxDim = Math.max(dimensions.x, dimensions.y, dimensions.z);

      if (centroid.z > 3.0) {
        // Airborne target
        if (maxDim <= 0.8) {
          classification = 'MICRO_DRONE';
        } else {
          classification = 'MEDIUM_UAV';
        }
      } else {
        // Ground target
        if (maxDim <= 2.0 && dimensions.z >= 1.2) {
          classification = 'PERSONNEL';
        } else if (maxDim > 2.0) {
          classification = 'VEHICLE';
        }
      }

      results.push({
        clusterId: cId,
        centroid,
        boundingBox: {
          min: { x: minX, y: minY, z: minZ },
          max: { x: maxX, y: maxY, z: maxZ },
          dimensions,
        },
        numPoints: count,
        classification,
      });
    }

    return results;
  }
}
