/**
 * Kinetic & Electronic Warfare (EW) C-UAS Effector Suite
 * 
 * Implements:
 * 1. High-Energy Laser (HEL) Directed Energy Dwell Calculator & Atmospheric Extinction
 * 2. Multi-Band RF / GNSS Jammer Link Budget & Jam-to-Signal (J/S) Solver
 * 3. 3D Augmented Proportional Navigation (APN) Interceptor Drone Guidance Law
 */

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export const Vec3 = {
  add: (a: Vector3D, b: Vector3D): Vector3D => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }),
  sub: (a: Vector3D, b: Vector3D): Vector3D => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }),
  scale: (v: Vector3D, s: number): Vector3D => ({ x: v.x * s, y: v.y * s, z: v.z * s }),
  dot: (a: Vector3D, b: Vector3D): number => a.x * b.x + a.y * b.y + a.z * b.z,
  cross: (a: Vector3D, b: Vector3D): Vector3D => ({
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  }),
  norm: (v: Vector3D): number => Math.hypot(v.x, v.y, v.z),
  normalize: (v: Vector3D): Vector3D => {
    const n = Math.hypot(v.x, v.y, v.z);
    return n > 1e-9 ? { x: v.x / n, y: v.y / n, z: v.z / n } : { x: 0, y: 0, z: 0 };
  },
};

// ==========================================
// 1. High-Energy Laser (HEL) Directed Energy
// ==========================================

export type TargetMaterial = 'CFRP_COMPOSITE' | 'ALUMINUM_AIRFRAME' | 'OPTICAL_SEEKER' | 'BATTERY_PACK';

export interface LaserSystemSpecs {
  powerWatts: number;          // Optical output power (e.g. 50,000 W = 50kW)
  wavelengthMeters: number;    // Beam wavelength (e.g. 1064e-9 for Nd:YAG / Fiber laser)
  apertureDiameterMeters: number; // Beam director mirror aperture (e.g. 0.35m)
  m2BeamQuality: number;       // M^2 factor (typically 1.1 - 1.3)
  jitterRmsRad: number;        // Gimbal beam stabilization jitter in radians (e.g. 3e-6 rad)
  atmosphericExtinctionPerKm: number; // Beer-Lambert attenuation (e.g. 0.08 /km)
}

export interface LaserDwellResult {
  rangeMeters: number;
  spotDiameterCm: number;
  irradianceWattsPerCm2: number;
  powerOnTargetWatts: number;
  requiredDwellTimeSec: number;
  isLethalAtRange: boolean;
  killMechanism: string;
}

export class HighEnergyLaserEffector {
  private specs: LaserSystemSpecs;

  // Material thermal ablation fluence thresholds (Joules / cm^2)
  private static readonly FLUENCE_THRESHOLDS: Record<TargetMaterial, { fluenceJPerCm2: number; absorptivity: number; mechanism: string }> = {
    CFRP_COMPOSITE: {
      fluenceJPerCm2: 12000,
      absorptivity: 0.88,
      mechanism: 'Epoxy resin vapor ignition & structural wing spar rupture',
    },
    ALUMINUM_AIRFRAME: {
      fluenceJPerCm2: 24000,
      absorptivity: 0.22,
      mechanism: 'Alloy skin melt-through & internal avionics thermal shock',
    },
    OPTICAL_SEEKER: {
      fluenceJPerCm2: 350,
      absorptivity: 0.95,
      mechanism: 'CMOS/InSb focal plane detector laser dazzle & sensor destruction',
    },
    BATTERY_PACK: {
      fluenceJPerCm2: 8500,
      absorptivity: 0.75,
      mechanism: 'Lithium-Polymer cell thermal runaway & catastrophic explosion',
    },
  };

  constructor(specs?: Partial<LaserSystemSpecs>) {
    this.specs = {
      powerWatts: specs?.powerWatts ?? 50000, // 50 kW standard
      wavelengthMeters: specs?.wavelengthMeters ?? 1064e-9,
      apertureDiameterMeters: specs?.apertureDiameterMeters ?? 0.35,
      m2BeamQuality: specs?.m2BeamQuality ?? 1.2,
      jitterRmsRad: specs?.jitterRmsRad ?? 3e-6,
      atmosphericExtinctionPerKm: specs?.atmosphericExtinctionPerKm ?? 0.08,
    };
  }

  /**
   * Calculates required dwell time and beam irradiance at a given target range
   */
  public calculateDwellTime(rangeMeters: number, material: TargetMaterial = 'CFRP_COMPOSITE'): LaserDwellResult {
    // 1. Diffraction-limited spot size: d_diff = 2.44 * lambda * R / D * M^2
    const dDiff = (2.44 * this.specs.wavelengthMeters * rangeMeters / this.specs.apertureDiameterMeters) * this.specs.m2BeamQuality;
    
    // 2. Jitter spot broadening: d_jitter = 2 * R * jitter_rms
    const dJitter = 2.0 * rangeMeters * this.specs.jitterRmsRad;

    // Total spot diameter in meters and cm
    const totalSpotDiameterMeters = Math.sqrt(dDiff * dDiff + dJitter * dJitter);
    const spotDiameterCm = totalSpotDiameterMeters * 100.0;
    const spotAreaCm2 = Math.PI * Math.pow(spotDiameterCm / 2.0, 2);

    // 3. Atmospheric transmission via Beer-Lambert law: P_target = P0 * exp(-gamma * R_km)
    const rangeKm = rangeMeters / 1000.0;
    const atmosphericTrans = Math.exp(-this.specs.atmosphericExtinctionPerKm * rangeKm);
    const powerOnTargetWatts = this.specs.powerWatts * atmosphericTrans;

    // Irradiance at focal spot (W/cm^2)
    const irradianceWattsPerCm2 = powerOnTargetWatts / Math.max(0.1, spotAreaCm2);

    // 4. Material dwell time calculation
    const mat = HighEnergyLaserEffector.FLUENCE_THRESHOLDS[material];
    const effectiveIrradiance = irradianceWattsPerCm2 * mat.absorptivity;
    const requiredDwellTimeSec = mat.fluenceJPerCm2 / Math.max(1.0, effectiveIrradiance);

    // Lethal range check: dwell time must be under 3.5 seconds for tactical base defense
    const isLethalAtRange = requiredDwellTimeSec <= 3.5 && rangeMeters <= 2500;

    return {
      rangeMeters,
      spotDiameterCm: Math.round(spotDiameterCm * 100) / 100,
      irradianceWattsPerCm2: Math.round(irradianceWattsPerCm2),
      powerOnTargetWatts: Math.round(powerOnTargetWatts),
      requiredDwellTimeSec: Math.round(requiredDwellTimeSec * 100) / 100,
      isLethalAtRange,
      killMechanism: mat.mechanism,
    };
  }
}

// ==========================================
// 2. RF / EW Jammer Link Budget (J/S Ratio)
// ==========================================

export type RFBand = 'GNSS_L1' | 'GNSS_L2' | 'ISM_2_4GHZ' | 'ISM_5_8GHZ' | 'UHF_433MHZ';

export interface RFBandConfig {
  frequencyHz: number;
  gcsTransmitPowerWatts: number;
  gcsAntennaGainDbi: number;
  jammerPowerWatts: number;
  jammerAntennaGainDbi: number;
  denialThresholdJsDb: number;
}

export interface JammerLinkBudgetResult {
  band: RFBand;
  droneRangeFromGcsMeters: number;
  droneRangeFromJammerMeters: number;
  signalPowerAtDroneDbm: number;
  jammerPowerAtDroneDbm: number;
  jamToSignalRatioDb: number;
  isLinkDenied: boolean;
  recommendedAction: string;
}

export class RFJammerEffector {
  private static readonly BAND_SPECS: Record<RFBand, RFBandConfig> = {
    GNSS_L1: {
      frequencyHz: 1575.42e6,
      gcsTransmitPowerWatts: 0.05, // Satellite received power equivalent (very weak, -130 dBm)
      gcsAntennaGainDbi: 0,
      jammerPowerWatts: 40.0,
      jammerAntennaGainDbi: 14.0,
      denialThresholdJsDb: 10.0,
    },
    GNSS_L2: {
      frequencyHz: 1227.60e6,
      gcsTransmitPowerWatts: 0.05,
      gcsAntennaGainDbi: 0,
      jammerPowerWatts: 40.0,
      jammerAntennaGainDbi: 14.0,
      denialThresholdJsDb: 10.0,
    },
    ISM_2_4GHZ: {
      frequencyHz: 2450.0e6,
      gcsTransmitPowerWatts: 1.0, // 1W Handheld GCS controller
      gcsAntennaGainDbi: 3.5,
      jammerPowerWatts: 100.0,    // 100W directional C-UAS amplifier
      jammerAntennaGainDbi: 16.0,
      denialThresholdJsDb: 15.0,
    },
    ISM_5_8GHZ: {
      frequencyHz: 5800.0e6,
      gcsTransmitPowerWatts: 1.0,
      gcsAntennaGainDbi: 4.0,
      jammerPowerWatts: 100.0,
      jammerAntennaGainDbi: 18.0,
      denialThresholdJsDb: 15.0,
    },
    UHF_433MHZ: {
      frequencyHz: 433.92e6,
      gcsTransmitPowerWatts: 0.5,
      gcsAntennaGainDbi: 2.15,
      jammerPowerWatts: 50.0,
      jammerAntennaGainDbi: 10.0,
      denialThresholdJsDb: 12.0,
    },
  };

  /**
   * Free Space Path Loss in dB: FSPL = 20*log10(d) + 20*log10(f) - 147.55
   */
  public static calculateFSPL(distanceMeters: number, frequencyHz: number): number {
    const d = Math.max(1.0, distanceMeters);
    return 20 * Math.log10(d) + 20 * Math.log10(frequencyHz) - 147.55;
  }

  /**
   * Calculate Jam-to-Signal (J/S) link budget for an enemy drone
   */
  public computeLinkBudget(
    band: RFBand,
    droneRangeFromJammerMeters: number,
    droneRangeFromGcsMeters = 1500
  ): JammerLinkBudgetResult {
    const spec = RFJammerEffector.BAND_SPECS[band];

    // GCS Signal power calculation:
    // P_rx_signal (dBm) = P_tx_gcs (dBm) + G_tx - FSPL
    const gcsPowerDbm = 10 * Math.log10(spec.gcsTransmitPowerWatts * 1000);
    const fsplSignal = RFJammerEffector.calculateFSPL(droneRangeFromGcsMeters, spec.frequencyHz);
    const signalPowerDbm = gcsPowerDbm + spec.gcsAntennaGainDbi - fsplSignal;

    // Jammer power calculation:
    // P_rx_jam (dBm) = P_tx_jam (dBm) + G_jam - FSPL - PolarizationLoss(3dB)
    const jammerPowerDbm = 10 * Math.log10(spec.jammerPowerWatts * 1000);
    const fsplJam = RFJammerEffector.calculateFSPL(droneRangeFromJammerMeters, spec.frequencyHz);
    const jammerPowerAtDroneDbm = jammerPowerDbm + spec.jammerAntennaGainDbi - fsplJam - 3.0;

    // Jam-to-Signal Ratio
    const jamToSignalRatioDb = jammerPowerAtDroneDbm - signalPowerDbm;
    const isLinkDenied = jamToSignalRatioDb >= spec.denialThresholdJsDb;

    let recommendedAction = 'RF LINK INTACT: Increase jammer gain or slew AESA antenna';
    if (isLinkDenied) {
      if (band.startsWith('GNSS')) {
        recommendedAction = 'GNSS DENIED: Target forced into sensor dead-reckoning drift';
      } else {
        recommendedAction = 'C2 LINK SEVERED: Target failsafe trigger (Forced Land / Lost-Link RTH)';
      }
    }

    return {
      band,
      droneRangeFromGcsMeters,
      droneRangeFromJammerMeters,
      signalPowerAtDroneDbm: Math.round(signalPowerDbm * 10) / 10,
      jammerPowerAtDroneDbm: Math.round(jammerPowerAtDroneDbm * 10) / 10,
      jamToSignalRatioDb: Math.round(jamToSignalRatioDb * 10) / 10,
      isLinkDenied,
      recommendedAction,
    };
  }
}

// ========================================================
// 3. 3D Augmented Proportional Navigation (APN) Guidance
// ========================================================

export interface GuidanceCommand {
  accelCmd: Vector3D;       // Commanded acceleration vector in m/s^2
  accelMagnitudeG: number;  // Commanded acceleration in G's
  closingVelocity: number;  // V_c in m/s
  timeToGoSec: number;      // Estimated time to intercept
  missDistanceEstimate: number; // Predicted closest point of approach in meters
}

export class ProportionalNavigationGuidance {
  private navConstant: number; // Typically N = 3 to 5
  private maxAccelG: number;   // Maximum airframe structural G limit (e.g. 15 G for C-UAS interceptor)

  constructor(navConstant = 4.0, maxAccelG = 15.0) {
    this.navConstant = navConstant;
    this.maxAccelG = maxAccelG;
  }

  /**
   * Computes 3D True Proportional Navigation acceleration command:
   * a_cmd = N * V_closing * (Omega_LOS x v_interceptor_hat)
   */
  public computeGuidance(
    interceptorPos: Vector3D,
    interceptorVel: Vector3D,
    targetPos: Vector3D,
    targetVel: Vector3D,
    targetAccel: Vector3D = { x: 0, y: 0, z: 0 }
  ): GuidanceCommand {
    // Relative position & velocity: R = r_tgt - r_int
    const rRel = Vec3.sub(targetPos, interceptorPos);
    const vRel = Vec3.sub(targetVel, interceptorVel);

    const range = Vec3.norm(rRel);
    if (range < 0.1) {
      return {
        accelCmd: { x: 0, y: 0, z: 0 },
        accelMagnitudeG: 0,
        closingVelocity: 0,
        timeToGoSec: 0,
        missDistanceEstimate: 0,
      };
    }

    // Line of sight unit vector
    const rHat = Vec3.normalize(rRel);

    // Closing velocity V_c = - (R . V_rel) / |R|
    const closingVelocity = -Vec3.dot(rRel, vRel) / range;
    const timeToGoSec = closingVelocity > 0 ? range / closingVelocity : range / 100.0;

    // Line of sight rotation rate vector: Omega = (R x V_rel) / |R|^2
    const crossProduct = Vec3.cross(rRel, vRel);
    const rSquared = range * range;
    const omegaLos = Vec3.scale(crossProduct, 1.0 / rSquared);

    // Interceptor velocity heading unit vector
    const vIntSpeed = Vec3.norm(interceptorVel);
    const vIntHat = vIntSpeed > 1e-3 ? Vec3.normalize(interceptorVel) : rHat;

    // Pure PN acceleration command: a_pn = N * V_c * (Omega x v_hat)
    const omegaCrossV = Vec3.cross(omegaLos, vIntHat);
    let accelCmd = Vec3.scale(omegaCrossV, this.navConstant * Math.max(10.0, closingVelocity));

    // Augmented PN (APN): Add target acceleration compensation: 0.5 * N * a_target_normal
    if (Vec3.norm(targetAccel) > 0.1) {
      const aTgtNormal = Vec3.sub(targetAccel, Vec3.scale(rHat, Vec3.dot(targetAccel, rHat)));
      const apnTerm = Vec3.scale(aTgtNormal, 0.5 * this.navConstant);
      accelCmd = Vec3.add(accelCmd, apnTerm);
    }

    // Enforce airframe G-limit
    const gAcc = 9.81;
    const accelMag = Vec3.norm(accelCmd);
    const maxAccelMs2 = this.maxAccelG * gAcc;

    if (accelMag > maxAccelMs2) {
      accelCmd = Vec3.scale(accelCmd, maxAccelMs2 / accelMag);
    }

    // Zero-Effort Miss (ZEM) estimate: ZEM = R + V_rel * t_go + 0.5 * a_tgt * t_go^2
    const zem = Vec3.add(rRel, Vec3.scale(vRel, timeToGoSec));
    const missDistance = Vec3.norm(zem);

    return {
      accelCmd,
      accelMagnitudeG: Vec3.norm(accelCmd) / gAcc,
      closingVelocity,
      timeToGoSec,
      missDistanceEstimate: Math.max(0, missDistance),
    };
  }

  /**
   * Advances simulation state of interceptor by time step dt
   */
  public stepSimulation(
    interceptorPos: Vector3D,
    interceptorVel: Vector3D,
    accelCmd: Vector3D,
    dt = 0.05
  ): { nextPos: Vector3D; nextVel: Vector3D } {
    const nextVel = Vec3.add(interceptorVel, Vec3.scale(accelCmd, dt));
    const nextPos = Vec3.add(interceptorPos, Vec3.scale(nextVel, dt));
    return { nextPos, nextVel };
  }
}
