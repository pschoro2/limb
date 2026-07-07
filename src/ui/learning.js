export const LEARNING_CONFIG = {
  recallThreshold: 0.9,
  riskWithdrawalThreshold: 0.5,
  safeValenceThreshold: 0.28,
  painThreshold: 0.5,
  warningPainThreshold: 0.3,
  depthClosingThreshold: 0.18,
  dodgeClosingSpeedThreshold: 10.0,
  contactStartDistance: 0.9,
  contactFullDistance: 0.22,
  learningContactThreshold: 0.14,
  cautiousProbeX: 0.2,
  closeObjectX: 1.15,
  learningRate: 0.42,
};

export class EmbodiedMemory {
  constructor(config = LEARNING_CONFIG, options = {}) {
    this.config = config;
    this.storageKey = options.storageKey ?? "limb.embodiedMemory.v1";
    this.storage = options.storage ?? globalThis.localStorage ?? null;
    this.entries = [];
    if (options.load !== false) this.loadFromStorage();
  }

  recall(cueVector) {
    const matches = this.entries
      .map((entry) => ({
        entry,
        similarity: cosineSimilarity(cueVector, entry.cueVector),
      }))
      .sort((left, right) => right.similarity - left.similarity);

    return {
      best: matches[0] ?? null,
      second: matches[1] ?? null,
      matches,
    };
  }

  trustedRecall(cueVector) {
    const recall = this.recall(cueVector);
    if (!recall.best || recall.best.similarity < this.config.recallThreshold) return null;
    return recall.best;
  }

  recordExperience({ cueVector, label, action, sensors, time, metadata = {} }) {
    const outcome = outcomeFromSensors(sensors, this.config);
    if (!outcome.learned) return null;

    const recall = this.recall(cueVector);
    const existing =
      recall.best && recall.best.similarity >= this.config.recallThreshold ? recall.best.entry : null;

    if (existing) {
      blendEntry(existing, {
        action,
        valence: outcome.valence,
        risk: outcome.risk,
        sensors,
        time,
        learningRate: this.config.learningRate,
        riskWithdrawalThreshold: this.config.riskWithdrawalThreshold,
      });
      existing.metadata = { ...existing.metadata, ...metadata };
      this.saveToStorage();
      return existing;
    }

    const entry = {
      id: globalThis.crypto?.randomUUID?.() ?? `${label}-${time.toFixed(3)}-${this.entries.length}`,
      label,
      cueVector: [...cueVector],
      action,
      valence: outcome.valence,
      risk: outcome.risk,
      strength: outcome.risk > this.config.riskWithdrawalThreshold ? 0.72 : 0.48,
      painPeak: sensors.pain,
      pressurePeak: sensors.pressure,
      contacts: 1,
      lastUsed: time,
      metadata,
    };
    this.entries.push(entry);
    this.saveToStorage();
    return entry;
  }

  reset() {
    this.entries = [];
    if (this.storage) this.storage.removeItem(this.storageKey);
  }

  toJSON() {
    return {
      version: 1,
      config: this.config,
      entries: this.entries.map((entry) => ({
        ...entry,
        cueVector: [...entry.cueVector],
      })),
    };
  }

  exportJSON() {
    return JSON.stringify(this.toJSON(), null, 2);
  }

  importEntries(entries) {
    this.entries = entries.map((entry) => normalizeEntry(entry));
    this.saveToStorage();
  }

  loadFromStorage() {
    if (!this.storage) return false;
    const raw = this.storage.getItem(this.storageKey);
    if (!raw) return false;
    try {
      const parsed = JSON.parse(raw);
      this.entries = Array.isArray(parsed.entries) ? parsed.entries.map((entry) => normalizeEntry(entry)) : [];
      return true;
    } catch {
      this.entries = [];
      return false;
    }
  }

  saveToStorage() {
    if (!this.storage) return false;
    this.storage.setItem(this.storageKey, this.exportJSON());
    return true;
  }
}

export function chooseAction({ itemX, cueVector, confidence, sensors, memory, config = LEARNING_CONFIG }) {
  if (!cueVector) {
    return decision("HOLD", "No object in stereo view.", confidence, 0, "unknown", null);
  }

  const currentRisk = Math.max(sensors.pain, sensors.puncture);
  if (sensors.pain >= config.painThreshold) {
    return decision(
      "WITHDRAW_FAST",
      "Fingertip pain reflex; storing this contact.",
      confidence,
      currentRisk,
      "danger",
      null,
    );
  }

  const recalled = memory.trustedRecall(cueVector);
  if (recalled && recalled.entry.risk >= config.riskWithdrawalThreshold) {
    return decision(
      recalled.entry.painPeak >= config.painThreshold || itemX > config.closeObjectX || sensors.approachingSensor
        ? "WITHDRAW_FAST"
        : "RETRACT",
      sensors.approachingSensor
        ? "Learned painful cue is closing on the fingertip sensors; maximum avoidance."
        : "Vision matched a learned painful cue; maximum avoidance reflex.",
      confidence,
      recalled.entry.risk,
      "danger",
      recalled,
    );
  }

  if (sensors.approachingSensor && itemX > config.cautiousProbeX) {
    return decision(
      "CAUTIOUS_PROBE",
      "Depth cue shows object closing on the fingertip sensors.",
      confidence,
      currentRisk,
      "unknown",
      null,
    );
  }

  if (recalled && recalled.entry.valence >= config.safeValenceThreshold) {
    return decision(
      itemX > config.closeObjectX ? "GENTLE_TOUCH" : "APPROACH_SLOW",
      "Learned low-pain contact; reaching cautiously.",
      confidence,
      recalled.entry.risk,
      "safe",
      recalled,
    );
  }

  if (itemX > config.cautiousProbeX) {
    return decision("CAUTIOUS_PROBE", "No trusted memory yet; probing through fingertip sensors.", confidence, currentRisk, "unknown", null);
  }

  return decision("ORIENT_CAMERA", "Stereo cue seen; waiting for contact evidence.", confidence, currentRisk, "unknown", null);
}

export function shouldDodgeReflex(decision, sensors, config = LEARNING_CONFIG) {
  return (
    decision?.action === "WITHDRAW_FAST" &&
    decision?.className === "danger" &&
    Boolean(decision?.recall) &&
    decision.recall.entry.painPeak >= config.painThreshold &&
    sensors.closingSpeed > config.dodgeClosingSpeedThreshold
  );
}

export function readTactileSensors({
  handPosition,
  objectPosition,
  sensorPoints = null,
  contactPoints = null,
  contactSamples = null,
  tactileProfile,
  previousDistance = null,
  deltaSeconds = 0,
  config = LEARNING_CONFIG,
}) {
  const sensors = sensorPoints?.length ? sensorPoints : [asPoint(handPosition, "sensor")];
  const contacts = contactPoints?.length ? contactPoints : [asPoint(objectPosition, "object")];
  const hasContactSamples = Array.isArray(contactSamples) && contactSamples.length > 0;
  const closest = hasContactSamples ? closestContactSample(contactSamples) : closestSensorContact(sensors, contacts);
  const distance = Number(closest.distance);
  const penetration = Number(closest.penetration ?? 0);
  const compressionDepth = Math.max(0.001, Number(closest.sensorRadius ?? 0.035));
  const closingSpeed =
    previousDistance !== null && deltaSeconds > 0 ? Math.max(0, (previousDistance - distance) / deltaSeconds) : 0;
  const contact = clamp01(
    (config.contactStartDistance - distance) / (config.contactStartDistance - config.contactFullDistance),
  );
  const tipSharpness = clamp01(tactileProfile.tipSharpness * (closest.contact.sharpnessScale ?? 1));
  const contactArea = clamp01(tactileProfile.contactArea * (closest.contact.areaScale ?? 1));
  const pressure = clamp01((penetration / compressionDepth) * (1.08 - tactileProfile.compliance * 0.35));
  const puncture = clamp01(
    pressure * (0.72 * tipSharpness + 0.28 * (1 - contactArea)),
  );
  const pain = clamp01(pressure * 0.28 + puncture * 0.82 - tactileProfile.compliance * 0.18);

  return {
    contact,
    pressure,
    puncture,
    pain,
    safeTouch: penetration > 0 && pain < config.painThreshold,
    painful: pain >= config.painThreshold,
    approachingSensor: closingSpeed >= config.depthClosingThreshold,
    closingSpeed,
    distance,
    sensorName: closest.sensor.name ?? "sensor",
    contactName: closest.contact.name ?? "object",
  };
}

export function estimateDepthFromDisparity(disparity, baseline = 0.18, focal = 1.8) {
  return disparity <= 0 ? Infinity : (baseline * focal) / disparity;
}

export function outcomeFromSensors(sensors, config = LEARNING_CONFIG) {
  if (sensors.contact < config.learningContactThreshold) {
    return { learned: false, valence: 0, risk: 0 };
  }

  if (sensors.pain >= config.painThreshold) {
    return {
      learned: true,
      valence: -Math.max(0.35, sensors.pain),
      risk: clamp01(Math.max(0.85, sensors.pain)),
    };
  }

  if (sensors.pain >= config.warningPainThreshold) {
    return {
      learned: true,
      valence: -Math.max(0.18, sensors.pain * 0.65),
      risk: clamp01(Math.max(0.35, sensors.pain + sensors.puncture * 0.35)),
    };
  }

  const comfort = clamp01(1 - sensors.pain - sensors.pressure * 0.18);
  return {
    learned: true,
    valence: clamp01(0.22 + comfort * 0.58),
    risk: clamp01(sensors.pain * 0.45 + sensors.puncture * 0.2),
  };
}

export function cosineSimilarity(left, right) {
  if (left.length !== right.length) {
    throw new Error(`Cue vector size mismatch: ${left.length} vs ${right.length}`);
  }

  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }

  const denom = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
  return denom === 0 ? 0 : clamp(dot / denom, -1, 1);
}

function blendEntry(entry, { action, valence, risk, sensors, time, learningRate, riskWithdrawalThreshold }) {
  const rate = risk >= riskWithdrawalThreshold ? Math.max(learningRate, 0.78) : learningRate;
  entry.valence = lerp(entry.valence, valence, rate);
  const blendedRisk = lerp(entry.risk, risk, rate);
  const accumulatedRisk = entry.risk + Math.max(0, risk) * 0.28;
  entry.risk = clamp01(Math.max(blendedRisk, accumulatedRisk));
  entry.strength = clamp01(entry.strength + 0.08 + risk * 0.18);
  entry.painPeak = Math.max(entry.painPeak, sensors.pain);
  entry.pressurePeak = Math.max(entry.pressurePeak, sensors.pressure);
  entry.contacts += 1;
  entry.action = action;
  entry.lastUsed = time;
}

function decision(action, reason, confidence, risk, className, recall) {
  return {
    action,
    reason,
    confidence: clamp01(confidence),
    risk: clamp01(risk),
    className,
    recall,
  };
}

function normalizeEntry(entry) {
  return {
    id: String(entry.id ?? globalThis.crypto?.randomUUID?.() ?? `memory-${Date.now()}`),
    label: String(entry.label ?? "unknown"),
    cueVector: Array.isArray(entry.cueVector) ? entry.cueVector.map(Number) : [],
    action: String(entry.action ?? "HOLD"),
    valence: Number(entry.valence ?? 0),
    risk: clamp01(entry.risk ?? 0),
    strength: clamp01(entry.strength ?? 0.5),
    painPeak: clamp01(entry.painPeak ?? 0),
    pressurePeak: clamp01(entry.pressurePeak ?? 0),
    contacts: Number(entry.contacts ?? 0),
    lastUsed: Number(entry.lastUsed ?? 0),
    metadata: typeof entry.metadata === "object" && entry.metadata !== null ? entry.metadata : {},
  };
}

function asPoint(point, name) {
  return {
    x: Number(point?.x ?? 0),
    y: Number(point?.y ?? 0),
    z: Number(point?.z ?? 0),
    name,
    sharpnessScale: 1,
    areaScale: 1,
  };
}

function closestSensorContact(sensors, contacts) {
  let best = {
    sensor: sensors[0],
    contact: contacts[0],
    distance: Infinity,
    penetration: 0,
    signedDistance: Infinity,
    sensorRadius: 0.035,
  };
  for (const sensor of sensors) {
    for (const contact of contacts) {
      const distance = Math.hypot(sensor.x - contact.x, sensor.y - contact.y, sensor.z - contact.z);
      if (distance < best.distance) {
        best = {
          sensor,
          contact,
          distance,
          penetration: 0,
          signedDistance: distance,
          sensorRadius: 0.035,
        };
      }
    }
  }
  return best;
}

function closestContactSample(contactSamples) {
  let best = {
    sensor: contactSamples[0]?.sensor ?? { x: 0, y: 0, z: 0, name: "sensor" },
    contact: contactSamples[0]?.contact ?? { x: 0, y: 0, z: 0, name: "object" },
    distance: Infinity,
    penetration: 0,
    signedDistance: Infinity,
    sensorRadius: 0.035,
  };
  for (const sample of contactSamples) {
    const distance = Number(sample?.distance ?? Infinity);
    const signedDistance = Number(sample?.signedDistance ?? distance);
    if (Number.isFinite(distance) && (distance < best.distance || signedDistance < best.signedDistance)) {
      best = {
        sensor: sample.sensor,
        contact: sample.contact,
        distance,
        penetration: Number(sample?.penetration ?? 0),
        signedDistance,
        sensorRadius: Number(sample?.sensorRadius ?? 0.035),
      };
    }
  }
  return best;
}

function lerp(left, right, amount) {
  return left + (right - left) * amount;
}

function clamp01(value) {
  return clamp(value, 0, 1);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, Number(value)));
}
