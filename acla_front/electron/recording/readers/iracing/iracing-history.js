'use strict';

const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const nonnegative = (value) => finite(value) && value >= 0;
const fraction = (value) => finite(value) && value >= 0 && value < 1;
const MAX_SAMPLE_GAP = 1;
const MAX_RELATIVE_SECONDS = 180;
const MAX_RELATIVE_SAMPLES = 1802;
const FUEL_EPSILON = 0.0001;

function trackLength(value) {
  const match = typeof value === 'string' && value.match(/^\s*(\d+(?:\.\d+)?)\s*(km|m)\s*$/i);
  return match ? Number(match[1]) * (match[2].toLowerCase() === 'km' ? 1000 : 1) : undefined;
}

function lapStep(before, after) {
  let step = after - before;
  if (step < -0.5) step += 1;
  if (step > 0.5) step -= 1;
  return step;
}

function crossingTime(history, position) {
  if (!history?.length) return undefined;
  const latest = history[history.length - 1];
  const target = latest.progress - ((latest.position - position + 1) % 1);
  for (let i = history.length - 1; i > 0; i -= 1) {
    const before = history[i - 1];
    const after = history[i];
    if (before.progress <= target && after.progress >= target && after.progress > before.progress) {
      return before.time + (after.time - before.time) * (target - before.progress) / (after.progress - before.progress);
    }
  }
  return undefined;
}

// Stateful calculations stay in the reader process, before the standard row is sent.
class IRacingHistory {
  constructor() { this.reset(); }

  reset() {
    this.previous = undefined;
    this.distance = undefined;
    this.usedFuel = undefined;
    this.lapFuel = undefined;
    this.consumption = [];
    this.sectorStart = undefined;
    this.lastSector = undefined;
    this.cars = new Map();
  }

  update(values, { length, sectors, drivers, playerIndex, fuelInLiters }, put) {
    const time = values.SessionTime;
    const position = values.LapDistPct;
    if (!nonnegative(time) || !fraction(position) || !(length > 0)
      || !nonnegative(values.Speed) || values.IsInGarage === true
      || typeof values.OnPitRoad !== 'boolean' || ![0, 1, 2, 3].includes(values.PlayerTrackSurface)
      || values.PlayerCarTowTime > 0) {
      this.reset();
      return;
    }
    const fuel = fuelInLiters && nonnegative(values.FuelLevel) ? values.FuelLevel : undefined;
    const pit = values.OnPitRoad === true;
    const stationaryPit = values.PlayerTrackSurface === 1 && values.Speed < 0.5;
    let previous = this.previous;
    let dt = previous ? time - previous.time : 0;
    let step = previous ? lapStep(previous.position, position) : 0;
    // Dropouts, towing/teleports, clock rewinds and lap resets break continuity.
    if (previous && (dt <= 0 || dt > MAX_SAMPLE_GAP
      || Math.abs(step) * length > 200 * dt + 5
      || (Number.isInteger(values.LapCompleted) && Number.isInteger(previous.laps)
        && (values.LapCompleted < previous.laps || values.LapCompleted > previous.laps + 1)))) {
      this.reset();
      previous = undefined;
      dt = 0;
      step = 0;
    }
    const refueling = previous && finite(fuel) && finite(previous.fuel) && fuel > previous.fuel + FUEL_EPSILON;
    if ((stationaryPit && !previous?.stationaryPit) || refueling) {
      this.distance = 0;
      this.usedFuel = finite(fuel) ? 0 : undefined;
      this.lapFuel = undefined;
      this.consumption = [];
      this.sectorStart = undefined;
      this.lastSector = undefined;
    } else if (previous) {
      if (finite(this.distance)) this.distance += (previous.speed + values.Speed) * dt / 2;
      if (finite(fuel) && finite(previous.fuel)) {
        if (finite(this.usedFuel)) this.usedFuel += Math.max(0, previous.fuel - fuel);
      } else {
        this.usedFuel = undefined;
        this.lapFuel = undefined;
        this.consumption = [];
      }
    }

    if (pit || step < 0 || values.PlayerTrackSurface === 1) {
      this.lapFuel = undefined;
      this.sectorStart = undefined;
      this.lastSector = undefined;
    } else if (previous && !previous.pit && !refueling && step > 0) {
      const boundaries = sectors.filter((sector) => sector.SectorStartPct > previous.position
        && sector.SectorStartPct <= previous.position + step)
        .map((sector) => sector.SectorStartPct);
      if (previous.position + step >= 1) {
        boundaries.push(1);
        boundaries.push(...sectors.filter((sector) => sector.SectorStartPct > 0 && sector.SectorStartPct <= position)
          .map((sector) => 1 + sector.SectorStartPct));
      }
      for (const boundary of boundaries.sort((a, b) => a - b)) {
        const ratio = (boundary - previous.position) / step;
        const crossedAt = previous.time + ratio * dt;
        if (finite(this.sectorStart)) this.lastSector = Math.round((crossedAt - this.sectorStart) * 1000);
        this.sectorStart = crossedAt;
        if (boundary === 1) {
          const crossedFuel = finite(fuel) && finite(previous.fuel) ? previous.fuel + ratio * (fuel - previous.fuel) : undefined;
          if (finite(crossedFuel) && finite(this.lapFuel) && this.lapFuel > crossedFuel + FUEL_EPSILON) {
            this.consumption.push(this.lapFuel - crossedFuel);
            if (this.consumption.length > 5) this.consumption.shift();
          }
          this.lapFuel = crossedFuel;
        }
      }
    }
    if (finite(this.distance)) put('Graphics_distance_traveled', this.distance);
    if (finite(fuel) && finite(this.usedFuel)) put('Graphics_used_fuel', this.usedFuel);
    if (finite(fuel) && this.consumption.length) {
      const perLap = this.consumption.reduce((total, value) => total + value, 0) / this.consumption.length;
      put('Graphics_fuel_per_lap', perLap);
      put('Graphics_fuel_estimated_laps', fuel / perLap);
    }
    if (Number.isInteger(this.lastSector) && sectors.length) {
      put('Graphics_last_sector_time', this.lastSector);
      // The shared catalog deliberately types both representations as integers.
      put('Graphics_last_sector_time_str', this.lastSector);
    }
    this.previous = { time, position, fuel, pit, stationaryPit, speed: values.Speed, laps: values.LapCompleted };
    this.updateGaps(values, { length, drivers, playerIndex }, put);
  }

  updateGaps(values, { length, drivers, playerIndex }, put) {
    const positions = values.CarIdxLapDistPct;
    const surfaces = values.CarIdxTrackSurface;
    const pits = values.CarIdxOnPitRoad;
    if (!Array.isArray(positions) || !Array.isArray(surfaces) || !Array.isArray(pits)) {
      this.cars.clear();
      return;
    }
    const active = new Map();
    for (const driver of drivers) {
      const index = driver.CarIdx;
      if (!Number.isInteger(index) || index < 0 || driver.CarIsPaceCar || driver.IsSpectator
        || !fraction(positions[index]) || ![0, 3].includes(surfaces[index]) || pits[index] !== false) continue;
      const position = positions[index];
      const identity = `${driver.UserID}:${driver.CarID}:${driver.CarNumber}`;
      const entry = this.cars.get(index);
      let history = entry?.identity === identity ? entry.history : [];
      const last = history[history.length - 1];
      const dt = last ? values.SessionTime - last.time : 0;
      const step = last ? lapStep(last.position, position) : 0;
      if (last && (dt <= 0 || dt > MAX_SAMPLE_GAP || step < 0 || step * length > 200 * dt + 5)) history = [];
      const progress = history.length ? last.progress + step : position;
      const current = { time: values.SessionTime, position, progress };
      // Keep 10 Hz history plus the latest sample; bounded by age and count.
      if (history.length > 1 && current.time - history[history.length - 2].time < 0.1) history.pop();
      history.push(current);
      while (history.length > MAX_RELATIVE_SAMPLES || (history.length > 1 && current.time - history[0].time > MAX_RELATIVE_SECONDS)) history.shift();
      this.cars.set(index, { identity, history });
      active.set(index, current);
    }
    for (const index of this.cars.keys()) if (!active.has(index)) this.cars.delete(index);
    const player = active.get(playerIndex);
    if (!player || values.OnPitRoad !== false) return;
    let ahead;
    let behind;
    for (const [index, car] of active) {
      if (index === playerIndex) continue;
      const forward = (car.position - player.position + 1) % 1;
      const backward = (player.position - car.position + 1) % 1;
      if (!ahead || forward < ahead.distance) ahead = { index, distance: forward };
      if (!behind || backward < behind.distance) behind = { index, distance: backward };
    }
    for (const [direction, neighbor] of [['ahead', ahead], ['behind', behind]]) {
      if (!neighbor) continue;
      const leader = direction === 'ahead' ? neighbor.index : playerIndex;
      const trailingPosition = direction === 'ahead' ? player.position : active.get(neighbor.index).position;
      const crossedAt = crossingTime(this.cars.get(leader)?.history, trailingPosition);
      if (finite(crossedAt)) put(`Graphics_gap_${direction}`, Math.round((values.SessionTime - crossedAt) * 1000));
    }
  }
}

module.exports = { IRacingHistory, trackLength };
