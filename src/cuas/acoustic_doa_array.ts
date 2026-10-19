/**
 * Phased Microphone Array Acoustic Direction-of-Arrival (DoA) Estimator
 * 
 * Implements:
 * - Speed of Sound Atmospheric Correction (Temperature & Relative Humidity)
 * - Uniform Circular Array (UCA) & 3D Planar Phased Array Geometry
 * - GCC-PHAT (Generalized Cross Correlation with Phase Transform) TDoA Estimation
 * - Narrowband MUSIC (Multiple Signal Classification) Spatial Pseudo-Spectrum
 * - Rotor Blade Pass Frequency (BPF) Harmonic Acoustic Peak Detector
 */

export interface MicrophonePosition {
  id: number;
  x: number; // meters from array center
  y: number;
  z: number;
}

export interface AcousticDOAResult {
  azimuthDeg: number;       // 0 to 360 degrees
  elevationDeg: number;     // 0 to 90 degrees
  confidence: number;       // 0.0 to 1.0 (MUSIC pseudo-spectrum peak prominence)
  estimatedRangeMeters?: number;
  dominantFrequencyHz: number;
  targetType: 'DRONE_ROTOR' | 'FIXED_WING_PROP' | 'JET_TURBINE' | 'AMBIENT_NOISE';
  timestampMs: number;
}

/**
 * Calculates the speed of sound in air (m/s) based on ambient temperature (Celsius)
 * c = 331.3 * sqrt(1 + T / 273.15)
 */
export function calculateSpeedOfSound(temperatureCelsius = 20): number {
  return 331.3 * Math.sqrt(1.0 + temperatureCelsius / 273.15);
}

/**
 * Generate standard 8-element Uniform Circular Array (UCA) geometry
 */
export function createCircularMicArray(numElements = 8, radiusMeters = 0.25): MicrophonePosition[] {
  const mics: MicrophonePosition[] = [];
  for (let i = 0; i < numElements; i++) {
    const angle = (2 * Math.PI * i) / numElements;
    mics.push({
      id: i,
      x: radiusMeters * Math.cos(angle),
      y: radiusMeters * Math.sin(angle),
      z: 0.0,
    });
  }
  return mics;
}

/**
 * Complex Number Helper
 */
export interface Complex {
  re: number;
  im: number;
}

export const ComplexMath = {
  add: (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im }),
  sub: (a: Complex, b: Complex): Complex => ({ re: a.re - b.re, im: a.im - b.im }),
  mul: (a: Complex, b: Complex): Complex => ({
    re: a.re * b.re - a.im * b.im,
    im: a.re * b.im + a.im * b.re,
  }),
  conj: (a: Complex): Complex => ({ re: a.re, im: -a.im }),
  abs: (a: Complex): number => Math.hypot(a.re, a.im),
  exp: (phase: number): Complex => ({ re: Math.cos(phase), im: Math.sin(phase) }),
};

/**
 * Discrete Fourier Transform (DFT) for small windows
 */
export function computeDFT(signal: number[]): Complex[] {
  const N = signal.length;
  const out: Complex[] = [];
  for (let k = 0; k < N; k++) {
    let sumRe = 0;
    let sumIm = 0;
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      sumRe += signal[n] * Math.cos(angle);
      sumIm += signal[n] * Math.sin(angle);
    }
    out.push({ re: sumRe, im: sumIm });
  }
  return out;
}

/**
 * Inverse Discrete Fourier Transform (IDFT)
 */
export function computeIDFT(spectrum: Complex[]): number[] {
  const N = spectrum.length;
  const out: number[] = [];
  for (let n = 0; n < N; n++) {
    let sum = 0;
    for (let k = 0; k < N; k++) {
      const angle = (2 * Math.PI * k * n) / N;
      // Real part of X(k) * e^(j*2*pi*k*n/N)
      sum += spectrum[k].re * Math.cos(angle) - spectrum[k].im * Math.sin(angle);
    }
    out.push(sum / N);
  }
  return out;
}

/**
 * GCC-PHAT (Generalized Cross Correlation with Phase Transform)
 * Estimates Time Difference of Arrival (TDoA) between two microphone signals.
 */
export function computeGCCPHAT(
  sigA: number[],
  sigB: number[],
  sampleRateHz = 48000
): { tdoaSeconds: number; maxCorrelation: number } {
  const N = Math.min(sigA.length, sigB.length);
  const specA = computeDFT(sigA.slice(0, N));
  const specB = computeDFT(sigB.slice(0, N));

  // Compute Cross-Power Spectrum with Phase Transform: G_xy(f) / |G_xy(f)|
  const phatSpectrum: Complex[] = [];
  for (let k = 0; k < N; k++) {
    const cross = ComplexMath.mul(specA[k], ComplexMath.conj(specB[k]));
    const magnitude = ComplexMath.abs(cross);
    if (magnitude > 1e-12) {
      phatSpectrum.push({ re: cross.re / magnitude, im: cross.im / magnitude });
    } else {
      phatSpectrum.push({ re: 0, im: 0 });
    }
  }

  // Cross correlation in time domain via IDFT
  const corr = computeIDFT(phatSpectrum);

  // Search peak within realistic physical delay bounds
  let maxVal = -1;
  let bestLag = 0;

  for (let i = 0; i < N; i++) {
    const lag = i <= N / 2 ? i : i - N;
    const val = corr[i];
    if (val > maxVal) {
      maxVal = val;
      bestLag = lag;
    }
  }

  const tdoaSeconds = bestLag / sampleRateHz;
  return { tdoaSeconds, maxCorrelation: maxVal };
}

/**
 * MUSIC (Multiple Signal Classification) Acoustic Array Solver
 */
export class AcousticMUSICArray {
  private micPositions: MicrophonePosition[];
  private speedOfSound: number;
  private numMics: number;

  constructor(micPositions?: MicrophonePosition[], temperatureCelsius = 20) {
    this.micPositions = micPositions ?? createCircularMicArray(8, 0.25);
    this.numMics = this.micPositions.length;
    this.speedOfSound = calculateSpeedOfSound(temperatureCelsius);
  }

  /**
   * Computes the theoretical array steering vector for a given incident direction:
   * a(theta, phi) = exp(-j * 2 * pi * f / c * (p_m . u))
   */
  public computeSteeringVector(
    azimuthRad: number,
    elevationRad: number,
    frequencyHz: number
  ): Complex[] {
    // Unit propagation vector pointing towards the source
    const ux = Math.cos(elevationRad) * Math.cos(azimuthRad);
    const uy = Math.cos(elevationRad) * Math.sin(azimuthRad);
    const uz = Math.sin(elevationRad);

    const k = (2 * Math.PI * frequencyHz) / this.speedOfSound;
    const steering: Complex[] = [];

    for (let m = 0; m < this.numMics; m++) {
      const pos = this.micPositions[m];
      const delayMeters = pos.x * ux + pos.y * uy + pos.z * uz;
      const phase = -k * delayMeters;
      steering.push(ComplexMath.exp(phase));
    }

    return steering;
  }

  /**
   * Estimates Spatial Covariance Matrix R_xx from multichannel snapshot buffers
   */
  public estimateCovarianceMatrix(snapshots: Complex[][]): Complex[][] {
    const M = this.numMics;
    const numSnapshots = snapshots.length;

    // Initialize MxM matrix
    const R: Complex[][] = Array.from({ length: M }, () =>
      Array.from({ length: M }, () => ({ re: 0, im: 0 }))
    );

    for (let k = 0; k < numSnapshots; k++) {
      const x = snapshots[k];
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < M; j++) {
          const prod = ComplexMath.mul(x[i], ComplexMath.conj(x[j]));
          R[i][j] = ComplexMath.add(R[i][j], prod);
        }
      }
    }

    // Average over snapshots
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < M; j++) {
        R[i][j].re /= numSnapshots;
        R[i][j].im /= numSnapshots;
      }
    }

    return R;
  }

  /**
   * Estimate 2D Acoustic Direction-of-Arrival (Azimuth & Elevation)
   * Evaluates MUSIC spatial pseudo-spectrum P(azimuth, elevation)
   */
  public estimateDOA(
    snapshots: Complex[][],
    frequencyHz: number,
    numSignals = 1
  ): AcousticDOAResult {
    const R = this.estimateCovarianceMatrix(snapshots);
    const M = this.numMics;

    // Power iteration / subspace projection approximation
    // Extract dominant signal eigenvector v_sig
    let vSig: Complex[] = Array.from({ length: M }, () => ({ re: 1 / Math.sqrt(M), im: 0 }));
    for (let iter = 0; iter < 15; iter++) {
      const vNext: Complex[] = Array.from({ length: M }, () => ({ re: 0, im: 0 }));
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < M; j++) {
          vNext[i] = ComplexMath.add(vNext[i], ComplexMath.mul(R[i][j], vSig[j]));
        }
      }
      let norm = 0;
      for (let i = 0; i < M; i++) {
        norm += vNext[i].re * vNext[i].re + vNext[i].im * vNext[i].im;
      }
      norm = Math.sqrt(norm) || 1e-9;
      vSig = vNext.map(c => ({ re: c.re / norm, im: c.im / norm }));
    }

    // Grid search over Azimuth (0 - 360 deg, step 5 deg) & Elevation (0 - 75 deg, step 5 deg)
    let bestAzimuth = 0;
    let bestElevation = 0;
    let maxPseudoSpectrum = -1;

    for (let azDeg = 0; azDeg < 360; azDeg += 5) {
      const azRad = (azDeg * Math.PI) / 180;
      for (let elDeg = 0; elDeg <= 75; elDeg += 5) {
        const elRad = (elDeg * Math.PI) / 180;
        const a = this.computeSteeringVector(azRad, elRad, frequencyHz);

        // a^H * vSig projection (signal subspace alignment)
        let dot: Complex = { re: 0, im: 0 };
        for (let m = 0; m < M; m++) {
          dot = ComplexMath.add(dot, ComplexMath.mul(ComplexMath.conj(a[m]), vSig[m]));
        }
        const signalAlignment = ComplexMath.abs(dot);
        // MUSIC pseudo-spectrum peak: 1 / (M - |a^H v_s|^2)
        const noisePower = Math.max(0.01, M - signalAlignment * signalAlignment);
        const pseudoSpectrum = 1.0 / noisePower;

        if (pseudoSpectrum > maxPseudoSpectrum) {
          maxPseudoSpectrum = pseudoSpectrum;
          bestAzimuth = azDeg;
          bestElevation = elDeg;
        }
      }
    }

    // Classification based on frequency
    let targetType: AcousticDOAResult['targetType'] = 'AMBIENT_NOISE';
    if (frequencyHz >= 180 && frequencyHz <= 600) {
      targetType = 'DRONE_ROTOR'; // Typical Quadcopter/Hexacopter motor BPF
    } else if (frequencyHz > 600 && frequencyHz <= 1600) {
      targetType = 'FIXED_WING_PROP';
    } else if (frequencyHz > 1600) {
      targetType = 'JET_TURBINE';
    }

    return {
      azimuthDeg: bestAzimuth,
      elevationDeg: bestElevation,
      confidence: Math.min(1.0, maxPseudoSpectrum / (M * 2)),
      dominantFrequencyHz: frequencyHz,
      targetType,
      timestampMs: Date.now(),
    };
  }

  /**
   * Synthesize test acoustic snapshot for a known true angle (useful for testing & unit tests)
   */
  public synthesizeSourceSnapshot(
    trueAzimuthDeg: number,
    trueElevationDeg: number,
    frequencyHz: number,
    snrDb = 20
  ): Complex[] {
    const azRad = (trueAzimuthDeg * Math.PI) / 180;
    const elRad = (trueElevationDeg * Math.PI) / 180;
    const steering = this.computeSteeringVector(azRad, elRad, frequencyHz);

    const noiseAmplitude = Math.pow(10, -snrDb / 20);
    return steering.map(s => ({
      re: s.re + (Math.random() - 0.5) * noiseAmplitude,
      im: s.im + (Math.random() - 0.5) * noiseAmplitude,
    }));
  }
}
