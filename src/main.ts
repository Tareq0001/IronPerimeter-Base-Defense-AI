/**
 * IronPerimeter: Autonomous Military Base Defense & C-UAS Digital Twin
 * Main Application Entrypoint
 * 
 * Modular TypeScript Integration:
 * - YOLOv8 + DeepSORT Tracker (src/vision/yolov8_deepsort_tracker.ts)
 * - Acoustic Phased Array MUSIC DoA (src/cuas/acoustic_doa_array.ts)
 * - Kinetic & EW Effectors: HEL Laser, RF Jammer, 3D ProNav (src/cuas/kinetic_ew_effector.ts)
 * - Multi-Tier Perimeter Geofence & LiDAR DBSCAN (src/geofence/perimeter_zone_engine.ts)
 */

import {
  DeepSortTracker,
  nonMaximumSuppression,
  BoundingBox,
  TrackedTarget,
} from './vision/yolov8_deepsort_tracker';
import {
  AcousticMUSICArray,
  createCircularMicArray,
  AcousticDOAResult,
} from './cuas/acoustic_doa_array';
import {
  HighEnergyLaserEffector,
  RFJammerEffector,
  ProportionalNavigationGuidance,
  Vec3,
  Vector3D,
} from './cuas/kinetic_ew_effector';
import {
  PerimeterGeofenceEngine,
  LiDARClusteringDBSCAN,
  LiDARPoint,
  ZoneEvaluationResult,
} from './geofence/perimeter_zone_engine';

// Declaration for global Three.js loaded via CDN
declare const THREE: any;

// ==========================================
// C-UAS & Base Defense Engine State
// ==========================================

export class IronPerimeterApp {
  // AI & Defense Modules
  public tracker = new DeepSortTracker({ minHits: 2, maxAge: 20 });
  public acousticArray = new AcousticMUSICArray(createCircularMicArray(8, 0.25), 22);
  public laserEffector = new HighEnergyLaserEffector({ powerWatts: 60000 });
  public rfJammer = new RFJammerEffector();
  public pnGuidance = new ProportionalNavigationGuidance(4.2, 16.0);
  public geofenceEngine = new PerimeterGeofenceEngine();
  public lidarDBSCAN = new LiDARClusteringDBSCAN(1.8, 4);

  // Three.js Scene Entities
  private scene: any;
  private camera: any;
  private renderer: any;
  private controls: any;
  private baseGroup: any;
  private radarBeams: any[] = [];
  private laserBeamMesh: any = null;
  private interceptorMesh: any = null;
  private targetDroneMeshes: any[] = [];

  // Interceptor Simulation State
  private interceptorActive = false;
  private interceptorPos: Vector3D = { x: 0, y: 0, z: 10 };
  private interceptorVel: Vector3D = { x: 0, y: 0, z: 0 };

  // Targets State
  private simulatedTargets: {
    id: number;
    pos: Vector3D;
    vel: Vector3D;
    type: 'UAV_ROTARY_QUAD' | 'LOITERING_MUNITION_FIXED_WING' | 'GROUND_INTRUDER';
    isJammed: boolean;
    isLaserTargeted: boolean;
    health: number; // 0 to 100
  }[] = [];

  // Active Alert & UI State
  private isBreached = false;
  private isAudioMuted = true;
  private audioCtx: AudioContext | null = null;
  private currentLang: 'en' | 'ar' = 'en';

  // Canvases
  private ptzCanvas: HTMLCanvasElement | null = null;
  private dasCanvas: HTMLCanvasElement | null = null;
  private ptzCtx: CanvasRenderingContext2D | null = null;
  private dasCtx: CanvasRenderingContext2D | null = null;
  private dasWavePhase = 0;

  constructor() {
    this.initDOM();
    this.initThreeScene();
    this.initSensorCanvases();
    this.initAudioContext();
    this.startSimulationLoop();
    this.logEvent('[SYSTEM]', 'IronPerimeter Modular Defense Engine Initialized.');
  }

  // ==========================================
  // DOM & UI Event Bindings
  // ==========================================

  private initDOM() {
    const breachBtn = document.getElementById('simulateBreachBtn');
    if (breachBtn) {
      breachBtn.addEventListener('click', () => this.toggleBreachSimulation());
    }

    const soundBtn = document.getElementById('soundBtn');
    if (soundBtn) {
      soundBtn.addEventListener('click', () => this.toggleAudio());
    }

    const resetCamBtn = document.getElementById('resetCamBtn');
    if (resetCamBtn) {
      resetCamBtn.addEventListener('click', () => this.resetCamera());
    }

    const langBtn = document.getElementById('langToggleBtn');
    if (langBtn) {
      langBtn.addEventListener('click', () => this.toggleLanguage());
    }

    const clearLogsBtn = document.getElementById('clearLogsBtn');
    if (clearLogsBtn) {
      clearLogsBtn.addEventListener('click', () => {
        const logList = document.getElementById('baseLogList');
        if (logList) logList.innerHTML = '';
      });
    }

    // Effector Quick Action Buttons if present
    const jamBtn = document.getElementById('jamBtn');
    if (jamBtn) {
      jamBtn.addEventListener('click', () => this.triggerJamming());
    }

    const laserBtn = document.getElementById('laserBtn');
    if (laserBtn) {
      laserBtn.addEventListener('click', () => this.triggerLaser());
    }

    const launchDroneBtn = document.getElementById('launchDroneBtn');
    if (launchDroneBtn) {
      launchDroneBtn.addEventListener('click', () => this.launchInterceptor());
    }
  }

  // ==========================================
  // Three.js 3D Perimeter Base Digital Twin
  // ==========================================

  private initThreeScene() {
    const container = document.getElementById('threeContainer');
    if (!container || typeof THREE === 'undefined') return;

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x030712);
    this.scene.fog = new THREE.FogExp2(0x030712, 0.0015);

    this.camera = new THREE.PerspectiveCamera(45, width / height, 1, 6000);
    this.camera.position.set(0, 280, 520);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);

    // Orbit Controls
    if ((THREE as any).OrbitControls) {
      this.controls = new (THREE as any).OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.04;
      this.controls.minDistance = 30;
      this.controls.maxDistance = 2500;
    }

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x1e293b, 1.2);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xfef3c7, 1.5);
    dirLight.position.set(300, 500, 200);
    dirLight.castShadow = true;
    this.scene.add(dirLight);

    // Defense Base Geometry
    this.baseGroup = new THREE.Group();
    this.buildBaseTerrain();
    this.buildPerimeterFences();
    this.buildWatchTowers();
    this.buildRadarAndEffectors();
    this.scene.add(this.baseGroup);

    // Resize Handler
    window.addEventListener('resize', () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
  }

  private buildBaseTerrain() {
    // Grid ground terrain
    const grid = new THREE.GridHelper(1200, 60, 0xd97706, 0x1e293b);
    grid.position.y = 0;
    this.baseGroup.add(grid);

    // Outer runway / tarmac
    const groundGeo = new THREE.PlaneGeometry(1600, 1600);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x050b14,
      roughness: 0.9,
      metalness: 0.1,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.5;
    ground.receiveShadow = true;
    this.baseGroup.add(ground);
  }

  private buildPerimeterFences() {
    // Concentric Security Rings (Geofences)
    const zones = this.geofenceEngine.getZones();
    for (const z of zones) {
      const radius = Math.hypot(z.polygon[0].x, z.polygon[0].y);
      const ringGeo = new THREE.RingGeometry(radius - 1.5, radius + 1.5, 64);
      const ringMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(z.colorHex),
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.45,
      });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.rotation.x = -Math.PI / 2;
      ringMesh.position.y = 0.5;
      this.baseGroup.add(ringMesh);
    }
  }

  private buildWatchTowers() {
    // 4 Corner Perimeter Watchtowers
    const corners = [
      { x: 180, z: 180 },
      { x: -180, z: 180 },
      { x: 180, z: -180 },
      { x: -180, z: -180 },
    ];

    const towerMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.6 });
    for (const c of corners) {
      const towerGeo = new THREE.CylinderGeometry(4, 7, 35, 8);
      const tower = new THREE.Mesh(towerGeo, towerMat);
      tower.position.set(c.x, 17.5, c.z);
      this.baseGroup.add(tower);

      // Searchlight / PTZ sphere
      const headGeo = new THREE.SphereGeometry(3.5, 8, 8);
      const headMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
      const head = new THREE.Mesh(headGeo, headMat);
      head.position.set(c.x, 36, c.z);
      this.baseGroup.add(head);
    }
  }

  private buildRadarAndEffectors() {
    // Central Radar Tower
    const radarTowerGeo = new THREE.CylinderGeometry(6, 10, 45, 12);
    const radarTowerMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8 });
    const radarTower = new THREE.Mesh(radarTowerGeo, radarTowerMat);
    radarTower.position.set(0, 22.5, 0);
    this.baseGroup.add(radarTower);

    // FMCW Rotating Radar Dish
    const dishGeo = new THREE.BoxGeometry(24, 6, 2);
    const dishMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });
    const dish = new THREE.Mesh(dishGeo, dishMat);
    dish.position.set(0, 48, 0);
    this.baseGroup.add(dish);
    this.radarBeams.push(dish);

    // Laser Beam Director
    const laserDirectorGeo = new THREE.SphereGeometry(5, 12, 12);
    const laserDirectorMat = new THREE.MeshStandardMaterial({ color: 0xef4444, metalness: 0.9 });
    const laserDirector = new THREE.Mesh(laserDirectorGeo, laserDirectorMat);
    laserDirector.position.set(0, 55, 0);
    this.baseGroup.add(laserDirector);

    // High Energy Laser Line Mesh
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 55, 0),
      new THREE.Vector3(0, 55, 0),
    ]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xff0044, linewidth: 3 });
    this.laserBeamMesh = new THREE.Line(lineGeo, lineMat);
    this.laserBeamMesh.visible = false;
    this.scene.add(this.laserBeamMesh);

    // Kinetic Interceptor Mesh
    const intGeo = new THREE.ConeGeometry(2, 6, 8);
    intGeo.rotateX(Math.PI / 2);
    const intMat = new THREE.MeshStandardMaterial({ color: 0x10b981, metalness: 0.8 });
    this.interceptorMesh = new THREE.Mesh(intGeo, intMat);
    this.interceptorMesh.visible = false;
    this.scene.add(this.interceptorMesh);
  }

  // ==========================================
  // Sensor Canvas Visualizations
  // ==========================================

  private initSensorCanvases() {
    this.ptzCanvas = document.getElementById('ptzCanvas') as HTMLCanvasElement;
    if (this.ptzCanvas) {
      this.ptzCtx = this.ptzCanvas.getContext('2d');
    }

    this.dasCanvas = document.getElementById('dasCanvas') as HTMLCanvasElement;
    if (this.dasCanvas) {
      this.dasCtx = this.dasCanvas.getContext('2d');
    }
  }

  private drawPTZThermalFeed() {
    if (!this.ptzCtx || !this.ptzCanvas) return;
    const ctx = this.ptzCtx;
    const w = this.ptzCanvas.width;
    const h = this.ptzCanvas.height;

    ctx.fillStyle = '#050c18';
    ctx.fillRect(0, 0, w, h);

    // Thermal noise grain
    for (let i = 0; i < 400; i++) {
      const rx = Math.random() * w;
      const ry = Math.random() * h;
      const intensity = Math.floor(Math.random() * 40);
      ctx.fillStyle = `rgb(${intensity}, ${intensity + 20}, ${intensity + 50})`;
      ctx.fillRect(rx, ry, 1, 1);
    }

    // Grid reticle
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Render target thermal heat blobs
    if (this.simulatedTargets.length > 0) {
      const tgt = this.simulatedTargets[0];
      const screenX = w / 2 + (tgt.pos.x / 10);
      const screenY = h / 2 - (tgt.pos.z / 10);

      // Thermal gradient
      const grad = ctx.createRadialGradient(screenX, screenY, 2, screenX, screenY, 18);
      grad.addColorStop(0, '#ffffff');
      grad.addColorStop(0.3, '#f59e0b');
      grad.addColorStop(0.7, '#ef4444');
      grad.addColorStop(1, 'rgba(239, 68, 68, 0)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(screenX, screenY, 18, 0, Math.PI * 2);
      ctx.fill();

      // Bounding box from AI Tracker
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(screenX - 12, screenY - 12, 24, 24);

      ctx.fillStyle = '#f59e0b';
      ctx.font = '9px monospace';
      ctx.fillText(`${tgt.type} [94%]`, screenX - 18, screenY - 16);
    }
  }

  private drawDASWaveform() {
    if (!this.dasCtx || !this.dasCanvas) return;
    const ctx = this.dasCtx;
    const w = this.dasCanvas.width;
    const h = this.dasCanvas.height;

    ctx.fillStyle = 'rgba(5, 11, 20, 0.35)';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = this.isBreached ? '#ef4444' : '#06b6d4';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    this.dasWavePhase += 0.08;
    const midY = h / 2;

    for (let x = 0; x < w; x++) {
      const freq = this.isBreached ? 0.08 : 0.03;
      const amplitude = this.isBreached ? 28 : 8;
      const noise = (Math.random() - 0.5) * (this.isBreached ? 12 : 3);
      const y = midY + Math.sin(x * freq + this.dasWavePhase) * amplitude + noise;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // ==========================================
  // Tactical Simulation Step
  // ==========================================

  private startSimulationLoop() {
    const loop = () => {
      requestAnimationFrame(loop);

      // Radar rotation
      for (const dish of this.radarBeams) {
        dish.rotation.y += 0.03;
      }

      // Update Target Physics
      this.updateTargets();

      // Update Interceptor Guidance
      this.updateInterceptor();

      // Update Canvas Feeds
      this.drawPTZThermalFeed();
      this.drawDASWaveform();

      if (this.controls) this.controls.update();
      if (this.renderer && this.scene && this.camera) {
        this.renderer.render(this.scene, this.camera);
      }
    };
    loop();
  }

  private updateTargets() {
    for (let i = this.simulatedTargets.length - 1; i >= 0; i--) {
      const tgt = this.simulatedTargets[i];

      // Move target towards center (base coordinates 0,0)
      if (!tgt.isJammed) {
        tgt.pos.x += tgt.vel.x * 0.05;
        tgt.pos.y += tgt.vel.y * 0.05;
        tgt.pos.z += tgt.vel.z * 0.05;
      } else {
        // Jammed drone drifts down slowly
        tgt.pos.y = Math.max(1, tgt.pos.y - 0.4);
      }

      // Evaluate against Geofence
      const evalResult: ZoneEvaluationResult = this.geofenceEngine.evaluateTrack(
        { x: tgt.pos.x, y: tgt.pos.z, z: tgt.pos.y },
        { x: tgt.vel.x, y: tgt.vel.z, z: tgt.vel.y }
      );

      this.updateHUDThreatStatus(tgt, evalResult);

      // Update 3D target mesh
      if (this.targetDroneMeshes[i]) {
        this.targetDroneMeshes[i].position.set(tgt.pos.x, tgt.pos.y, tgt.pos.z);
      }

      // Laser damage burn
      if (tgt.isLaserTargeted) {
        tgt.health -= 1.8;
        if (tgt.health <= 0) {
          this.logEvent('[DEW KILL]', `Target #${tgt.id} destroyed via High Energy Laser ablation.`);
          this.destroyTarget(i);
          continue;
        }
      }
    }
  }

  private updateInterceptor() {
    if (!this.interceptorActive || this.simulatedTargets.length === 0) {
      if (this.interceptorMesh) this.interceptorMesh.visible = false;
      return;
    }

    const tgt = this.simulatedTargets[0];
    const guidance = this.pnGuidance.computeGuidance(
      this.interceptorPos,
      this.interceptorVel,
      tgt.pos,
      tgt.vel
    );

    const step = this.pnGuidance.stepSimulation(
      this.interceptorPos,
      this.interceptorVel,
      guidance.accelCmd,
      0.05
    );

    this.interceptorPos = step.nextPos;
    this.interceptorVel = step.nextVel;

    if (this.interceptorMesh) {
      this.interceptorMesh.visible = true;
      this.interceptorMesh.position.set(this.interceptorPos.x, this.interceptorPos.y, this.interceptorPos.z);
      this.interceptorMesh.lookAt(tgt.pos.x, tgt.pos.y, tgt.pos.z);
    }

    // Intercept hit check (distance < 5m)
    const dist = Vec3.norm(Vec3.sub(tgt.pos, this.interceptorPos));
    if (dist < 8.0) {
      this.logEvent('[KINETIC HIT]', `Proportional Navigation Interceptor neutralized Target #${tgt.id}!`);
      this.interceptorActive = false;
      this.destroyTarget(0);
    }
  }

  private destroyTarget(idx: number) {
    if (this.targetDroneMeshes[idx]) {
      this.scene.remove(this.targetDroneMeshes[idx]);
      this.targetDroneMeshes.splice(idx, 1);
    }
    this.simulatedTargets.splice(idx, 1);
    if (this.laserBeamMesh) this.laserBeamMesh.visible = false;
    this.isBreached = this.simulatedTargets.length > 0;
  }

  private updateHUDThreatStatus(tgt: any, evalResult: ZoneEvaluationResult) {
    const classElem = document.getElementById('aiTargetClass');
    if (classElem) {
      classElem.innerText = `${tgt.type} (ID #${tgt.id})`;
      classElem.className = evalResult.threatLevel === 'CRITICAL_LETHAL' ? 'text-rose-400 font-bold' : 'text-amber-400 font-bold';
    }

    const confElem = document.getElementById('aiConfidence');
    if (confElem) confElem.innerText = '96.4%';

    const actionElem = document.getElementById('aiAction');
    if (actionElem) actionElem.innerText = evalResult.recommendedIntervention;

    const telIntruder = document.getElementById('telIntruder');
    if (telIntruder) {
      telIntruder.innerText = `${evalResult.threatLevel}: DIST ${evalResult.distanceToBoundaryMeters}m`;
      telIntruder.className = evalResult.threatLevel === 'CRITICAL_LETHAL' ? 'font-bold text-rose-500 animate-pulse' : 'font-bold text-amber-400';
    }
  }

  // ==========================================
  // Public Interactive Controls
  // ==========================================

  public toggleBreachSimulation() {
    this.isBreached = !this.isBreached;
    const btn = document.getElementById('simulateBreachBtn');
    const txt = document.getElementById('txtBreachBtn');

    if (this.isBreached) {
      if (txt) txt.innerText = this.currentLang === 'ar' ? 'إنهاء التهديد' : 'CLEAR INTRUSION';
      if (btn) btn.className = 'px-2.5 py-1 text-xs font-bold rounded bg-rose-600 text-white transition flex items-center space-x-1 shadow-lg shadow-rose-600/40';

      // Spawn Hostile Drone Inbound
      const newTarget = {
        id: Math.floor(Math.random() * 900) + 100,
        pos: { x: 450, y: 80, z: 350 },
        vel: { x: -35, y: -2, z: -28 },
        type: 'LOITERING_MUNITION_FIXED_WING' as const,
        isJammed: false,
        isLaserTargeted: false,
        health: 100,
      };
      this.simulatedTargets.push(newTarget);

      // Create 3D Mesh
      const droneGeo = new THREE.ConeGeometry(3, 8, 8);
      droneGeo.rotateX(Math.PI / 2);
      const droneMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
      const droneMesh = new THREE.Mesh(droneGeo, droneMat);
      droneMesh.position.set(newTarget.pos.x, newTarget.pos.y, newTarget.pos.z);
      this.scene.add(droneMesh);
      this.targetDroneMeshes.push(droneMesh);

      this.playSiren();
      this.logEvent('[BREACH ALERT]', `FMCW Radar Track Locked: Inbound Fixed-Wing Loitering Munition.`);
    } else {
      if (txt) txt.innerText = this.currentLang === 'ar' ? 'محاكاة محاولة اختراق' : 'INJECT BREACH';
      if (btn) btn.className = 'px-2.5 py-1 text-xs font-bold rounded bg-rose-950 border border-rose-600/60 hover:bg-rose-900 text-rose-300 transition flex items-center space-x-1';

      while (this.simulatedTargets.length > 0) {
        this.destroyTarget(0);
      }
      this.logEvent('[STAND-DOWN]', 'Perimeter cleared. Security baseline restored.');
    }
  }

  public triggerJamming() {
    if (this.simulatedTargets.length === 0) return;
    const tgt = this.simulatedTargets[0];
    const budget = this.rfJammer.computeLinkBudget('ISM_2_4GHZ', 600, 1800);

    tgt.isJammed = true;
    this.logEvent('[EW JAMMER]', `AESA RF Jammer Active: J/S = ${budget.jamToSignalRatioDb} dB. Target C2 severed.`);
  }

  public triggerLaser() {
    if (this.simulatedTargets.length === 0) return;
    const tgt = this.simulatedTargets[0];
    const dwell = this.laserEffector.calculateDwellTime(800, 'CFRP_COMPOSITE');

    tgt.isLaserTargeted = true;
    if (this.laserBeamMesh) {
      this.laserBeamMesh.visible = true;
      const positions = this.laserBeamMesh.geometry.attributes.position.array;
      positions[0] = 0; positions[1] = 55; positions[2] = 0;
      positions[3] = tgt.pos.x; positions[4] = tgt.pos.y; positions[5] = tgt.pos.z;
      this.laserBeamMesh.geometry.attributes.position.needsUpdate = true;
    }

    this.logEvent('[DIRECTED ENERGY]', `HEL 60kW Dwell Firing: Irradiance ${dwell.irradianceWattsPerCm2} W/cm², Est. Dwell: ${dwell.requiredDwellTimeSec}s.`);
  }

  public launchInterceptor() {
    if (this.simulatedTargets.length === 0) return;
    this.interceptorActive = true;
    this.interceptorPos = { x: 0, y: 25, z: 0 };
    this.interceptorVel = { x: 60, y: 20, z: 60 };
    this.logEvent('[KINETIC LAUNCH]', 'Autonomous C-UAS Interceptor Launched with 3D Augmented Proportional Navigation.');
  }

  private initAudioContext() {
    window.addEventListener('click', () => {
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
    }, { once: true });
  }

  public toggleAudio() {
    this.isAudioMuted = !this.isAudioMuted;
    const icon = document.getElementById('soundIcon');
    if (icon) {
      icon.className = this.isAudioMuted ? 'fa-solid fa-volume-xmark text-slate-500' : 'fa-solid fa-volume-high text-amber-400';
    }
  }

  private playSiren() {
    if (this.isAudioMuted || !this.audioCtx) return;
    const now = this.audioCtx.currentTime;
    const osc = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(650, now);
    osc.frequency.linearRampToValueAtTime(1150, now + 0.35);
    osc.frequency.linearRampToValueAtTime(650, now + 0.7);
    gain.gain.setValueAtTime(0.18, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.9);
    osc.connect(gain);
    gain.connect(this.audioCtx.destination);
    osc.start(now);
    osc.stop(now + 0.9);
  }

  public resetCamera() {
    if (this.camera && this.controls) {
      this.camera.position.set(0, 280, 520);
      this.controls.target.set(0, 0, 0);
      this.controls.update();
    }
  }

  public toggleLanguage() {
    this.currentLang = this.currentLang === 'en' ? 'ar' : 'en';
    const langBtn = document.getElementById('langToggleBtn');
    if (langBtn) {
      langBtn.innerText = this.currentLang === 'en' ? 'عربي' : 'English';
    }
    const htmlTag = document.documentElement;
    htmlTag.setAttribute('dir', this.currentLang === 'ar' ? 'rtl' : 'ltr');
  }

  public logEvent(tag: string, message: string) {
    const list = document.getElementById('baseLogList');
    if (!list) return;

    const timeStr = new Date().toTimeString().split(' ')[0];
    const div = document.createElement('div');
    div.className = 'p-1 rounded bg-black/40 border-l-2 border-amber-500';
    div.innerHTML = `<span class="text-slate-500">${timeStr}</span> <span class="text-amber-400 font-bold">${tag}</span> ${message}`;

    list.insertBefore(div, list.firstChild);
    while (list.children.length > 20) {
      list.removeChild(list.lastChild!);
    }
  }
}

// Instantiate on window load
window.addEventListener('DOMContentLoaded', () => {
  (window as any).ironPerimeterApp = new IronPerimeterApp();
});
