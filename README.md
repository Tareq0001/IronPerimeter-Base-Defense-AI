# IronPerimeter — Military Base Defense & AI Perimeter Security Twin

[![Live Demo](https://img.shields.io/badge/Live_Perimeter_Twin-Tareq0001.github.io-amber?style=for-the-badge&logo=github)](https://tareq0001.github.io/IronPerimeter-Base-Defense-AI/)
[![Sensors](https://img.shields.io/badge/Sensor_Fusion-Fiber_DAS_%7C_FMCW_Radar-blue?style=for-the-badge)](https://tareq0001.github.io/IronPerimeter-Base-Defense-AI/)
[![AI Vision](https://img.shields.io/badge/Edge_AI-Slew--to--Cue_PTZ_%7C_YOLO_Defense-orange?style=for-the-badge)](https://tareq0001.github.io/IronPerimeter-Base-Defense-AI/)

> **A mission-critical physical security operations center and digital twin for critical military infrastructure defense. Integrates fiber-optic Distributed Acoustic Sensing (DAS) along perimeter fence lines, FMCW ground surveillance radar, automated Slew-to-Cue Pan-Tilt-Zoom (PTZ) thermal camera targeting, and Edge AI threat classification to eliminate false alarms and neutralize intrusions.**

---

## ⚡ Core Physical Security & AI Sensor Capabilities

1. **3D High-Security Military Base & Perimeter Twin**
   - High-fidelity Three.js 3D facility simulation with guard towers, Entry Control Point (ECP) vehicle barriers, double security fence, and central command mast.
   - Interactive sector monitoring (Sectors 1-A through 4-B).

2. **Fiber-Optic Distributed Acoustic Sensing (DAS)**
   - Real-time optical time-domain reflectometry (OTDR) acoustic waveform display sampled at 10 kHz.
   - Distinguishes mechanical fence cutting, climbing vibrations, and underground tunneling from benign environmental noise.

3. **Automated Slew-to-Cue PTZ Thermal Camera Tracking**
   - Autonomous camera slewing: upon radar or acoustic disturbance detection, the PTZ camera instantly pans and tilts to the precise breach azimuth and elevation.
   - Real-time thermal shader simulation with targeting crosshairs.

4. **Edge AI Computer Vision Threat Classification**
   - Neural network model classifying perimeter detections:
     - `[HUMAN]: INTRUDER CLIMBING FENCE (97.8% Confidence)`
     - `[VEHICLE]: APPROACHING UNCHECKED (99.2% Confidence)`
     - `[ANIMAL]: WILD BOAR / STRAY (False Alarm Suppressed 96.5%)`
   - Automated rapid reaction force (QRF) alert and base lockdown protocols.

---

## 🛠️ Architecture & Tech Stack

- **Graphics:** Three.js (r128), Custom thermal shader canvas, DAS vibration waveform.
- **Audio:** Web Audio API security siren alarm synthesis.
- **Sensors:** FMCW Radar equations, Fiber-Optic OTDR vibration modeling, PTZ Slew-to-Cue geometry.
- **UI:** Tactile glassmorphism physical security deck, bilingual English/Arabic interface.

---

## 🚀 Live Deployment

- **Online Base Defense Center:** [https://tareq0001.github.io/IronPerimeter-Base-Defense-AI/](https://tareq0001.github.io/IronPerimeter-Base-Defense-AI/)
- **Author:** Eng. Tareq Aboushi (Tareq0001)
