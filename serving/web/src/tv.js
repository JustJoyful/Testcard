// TV, VCR and tape: the whole interface. Renders central state; holds a visual state only for its
// own animation (<= 800 ms). No backend calls here. Screen text duplicates the live region, so it is
// aria-hidden; only the on-screen menu buttons are exposed.
import { middleEllipsis, clockText, scoreText, secondsText } from './format.js';

const TEMPLATE = `
<div class="tv">
  <div class="tv-bezel">
    <div class="screen">
      <div class="screen-art" aria-hidden="true">
        <canvas class="noise" width="160" height="120"></canvas>
        <div class="screen-color"></div>
        <div class="tracking"></div>
        <div class="osd">
          <p class="osd-corner osd-tl"></p>
          <p class="osd-corner osd-br"></p>
          <div class="osd-center">
            <p class="osd-main"><span class="osd-main-text"></span><span class="osd-cursor"></span></p>
            <p class="osd-sub"></p>
            <p class="osd-bar"><span class="cells"></span><span class="osd-pct"></span></p>
            <div class="osd-score">
              <p class="osd-score-value"></p>
              <p class="osd-score-scale">/ 5</p>
              <p class="osd-range">OUTSIDE TRAINED RANGE 1–5</p>
              <p class="osd-meta"></p>
            </div>
          </div>
        </div>
      </div>
      <div class="osd-menu" role="toolbar" aria-label="Result">
        <button type="button" class="osd-item" data-act="details" aria-label="Show details" aria-expanded="false" aria-controls="details">[ DETAILS ]</button>
        <button type="button" class="osd-item" data-act="json" aria-label="Show JSON" aria-expanded="false" aria-controls="details" tabindex="-1">[ JSON ]</button>
        <button type="button" class="osd-item" data-act="eject" aria-label="Eject and try another video" tabindex="-1">[ EJECT ]</button>
      </div>
      <div class="screen-glass" aria-hidden="true">
        <div class="scanlines"></div>
        <div class="glass"></div>
        <div class="power-on"></div>
      </div>
    </div>
  </div>
  <div class="tv-panel" aria-hidden="true">
    <span class="knob"></span><span class="knob"></span>
    <span class="grille"></span>
  </div>
</div>
<div class="deck">
  <div class="vcr">
    <div class="vcr-slot"><span class="vcr-flap"></span></div>
    <button type="button" class="vcr-eject" aria-label="Eject" disabled>Eject</button>
    <div class="vcr-display" aria-hidden="true"><span class="vfd-ghost">88:88:88</span><span class="vfd-text"></span></div>
    <span class="vcr-led"></span>
  </div>
  <span class="tape-shadow" aria-hidden="true"></span>
  <button class="tape" type="button">
    <span class="sr-only">Choose a video: </span>
    <span class="tape-label">
      <span class="label-empty label-drag">DRAG A VIDEO HERE</span>
      <span class="label-empty label-tap">TAP TO CHOOSE A VIDEO</span>
      <span class="label-file"></span>
    </span>
    <span class="tape-window" aria-hidden="true"><span class="reel"></span><span class="reel"></span></span>
  </button>
</div>`;

const INSIDE = new Set(['inserting', 'uploading', 'queued', 'processing', 'done']);
const NOISY = new Set(['idle', 'dragover', 'invalid', 'inserting']);
const STAGE = { load: 'WARMING UP', extract: 'READING FRAMES', infer: 'SCORING' };
const MAIN = {
  idle: 'INSERT TAPE', standby: 'STANDBY', dragover: 'RELEASE TO LOAD', invalid: 'TAPE REJECTED',
  inserting: 'LOADING', uploading: 'LOADING', queued: 'WAITING', error: 'NO SIGNAL',
};
const CORNER = { idle: 'CH 03', dragover: 'CH 03', queued: '▶ PLAY', processing: '▶ PLAY', done: '■ STOP' };
// One short line per failure; the full sentence is in the page's status line and the live region.
const REASON = {
  unsupported_type: 'MP4 · MOV · WEBM ONLY',
  file_too_large: 'OVER 100 MIB',
  invalid_video: "CAN'T READ THIS VIDEO",
  busy: 'BUSY · TRY AGAIN SOON',
  upload_timeout: 'UPLOAD TIMED OUT',
  invalid_upload: 'UPLOAD INCOMPLETE',
  task_not_found: 'RESULT EXPIRED',
  timeout: 'TOOK TOO LONG',
  network: 'CONNECTION LOST',
  offline: 'TRY AGAIN LATER',
};
const HOLD = { insert: 600, eject: 500, shake: 300, burst: 400 };
const FRAME_MS = 1000 / 24;
const CELLS = 10;

export function mountTv(root, { chooseFile, onMenu, onEject, startedAt = () => null }) {
  root.innerHTML = TEMPLATE; // static markup only; every dynamic value below goes through textContent
  const $ = (s) => root.querySelector(s);
  const canvas = $('.noise');
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(160, 120);
  const pixels = new Uint32Array(image.data.buffer);
  const tape = $('.tape');
  const eject = $('.vcr-eject');
  const vfd = $('.vfd-text');
  const items = [...root.querySelectorAll('.osd-item')];
  const cells = $('.cells');
  cells.append(...Array.from({ length: CELLS }, () => document.createElement('i')));
  const motion = matchMedia('(prefers-reduced-motion: reduce)');

  let latest = null;
  let view = null; // the state name currently drawn
  let tapePos = 'rest'; // rest | inserting | inside | ejecting
  let tapeFile = null;
  let holdUntil = 0;
  let holdTimer = 0;
  let phase = ''; // 'burst' during the done static burst
  let focusedFor = null; // the result whose menu has already taken focus
  let raf = 0;
  let lastFrame = 0;
  let seed = 0x9e3779b9;
  let clockTimer = 0;
  let disposed = false;
  canvas.dataset.frames = '0';

  tape.addEventListener('click', () => chooseFile());
  eject.addEventListener('click', () => onEject());

  // On-screen menu: a toolbar with one tab stop; arrows move the highlight, Enter activates.
  const pick = (i, focus = true) => {
    items.forEach((b, j) => { b.tabIndex = j === i ? 0 : -1; });
    if (focus) items[i].focus();
  };
  items.forEach((b, i) => {
    b.addEventListener('click', () => {
      pick(i, false);
      if (b.dataset.act === 'eject') onEject();
      else onMenu(b.dataset.act);
    });
    b.addEventListener('pointerenter', () => pick(i, false));
  });
  $('.osd-menu').addEventListener('keydown', (e) => {
    const i = items.indexOf(document.activeElement);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: items.length - 1 }[e.key];
    if (i < 0 || next === undefined) return;
    e.preventDefault();
    pick((next + items.length) % items.length);
  });

  function drawNoise() {
    for (let i = 0; i < pixels.length; i += 1) {
      seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
      const v = (seed >>> 24) & 0xff;
      pixels[i] = 0xff000000 | (v << 16) | (v << 8) | v;
    }
    ctx.putImageData(image, 0, 0);
    canvas.dataset.frames = String(Number(canvas.dataset.frames) + 1);
  }

  const noiseWanted = () => !disposed && (NOISY.has(view) || phase === 'burst');

  function loop(now) {
    raf = 0;
    if (!noiseWanted() || document.hidden || motion.matches) return;
    if (now - lastFrame >= FRAME_MS) {
      lastFrame = now;
      drawNoise();
    }
    raf = requestAnimationFrame(loop);
  }

  function syncNoise() {
    if (!noiseWanted() || document.hidden || motion.matches) {
      cancelAnimationFrame(raf);
      raf = 0;
      // Reduced motion: one still frame, never animated.
      if (noiseWanted() && motion.matches && canvas.dataset.frames === '0') drawNoise();
      return;
    }
    if (!raf) raf = requestAnimationFrame(loop);
  }

  const elapsed = () => {
    const t0 = startedAt();
    return clockText(t0 === null ? 0 : performance.now() - t0);
  };

  function paintClocks() {
    let text = '12:00';
    if (view === 'standby') text = '';
    else if (view === 'invalid' || view === 'error') text = 'ERR';
    else if (view === 'inserting') text = 'LOAD';
    else if (view === 'uploading') text = `UP ${String(latest.uploadPct).padStart(2, ' ')}`;
    else if (view === 'queued' || view === 'processing') text = elapsed();
    else if (view === 'done') text = 'END';
    vfd.textContent = text;
    $('.osd-br').textContent = view === 'queued' || view === 'processing' ? elapsed() : '';
  }

  function syncClock() {
    const running = view === 'queued' || view === 'processing';
    if (running && !clockTimer) clockTimer = setInterval(paintClocks, 250);
    if (!running && clockTimer) {
      clearInterval(clockTimer);
      clockTimer = 0;
    }
  }

  function hold(ms) {
    if (motion.matches) return;
    holdUntil = performance.now() + ms;
    clearTimeout(holdTimer);
    holdTimer = setTimeout(() => {
      holdUntil = 0;
      phase = '';
      if (!disposed) apply(latest);
    }, ms);
  }

  function setTape(pos) {
    tapePos = pos;
    root.dataset.tape = pos;
    if (pos === 'rest') {
      tapeAnim?.cancel(); // hand the resting pose back to CSS (hover, drag-over, shake)
      tapeAnim = null;
    }
  }

  function paintLabel(file) {
    root.dataset.label = file ? 'file' : 'empty';
    $('.label-file').textContent = file ? middleEllipsis(file.name, 22) : '';
  }

  // The insert path, measured from the real layout: the tape lifts, travels to the slot and squares
  // up, then sinks behind the slot's top edge (clipped there), as if the VCR swallows it.
  // Eject plays the same path backwards. Transform origin is the tape's bottom centre.
  let tapeAnim = null;
  function playTape(kind) {
    tapeAnim?.cancel();
    const deck = tape.offsetParent.getBoundingClientRect();
    const slot = $('.vcr-slot').getBoundingClientRect();
    const w = tape.offsetWidth;
    const h = tape.offsetHeight;
    const dx = slot.left + slot.width / 2 - (deck.left + tape.offsetLeft + w / 2);
    const dy = slot.top + 1 - (deck.top + tape.offsetTop + h);
    const s = Math.min(1, (slot.width * 0.92) / w);
    const pose = (x, y, deg, scale, clip) => ({
      transform: `translate(${x}px, ${y}px) rotate(${deg}deg) scale(${scale})`,
      clipPath: `inset(0 0 ${clip}% 0)`,
    });
    tapeAnim = tape.animate([
      { ...pose(0, 0, -6, 1, 0), offset: 0, easing: 'cubic-bezier(.3, 0, .2, 1)' },
      { ...pose(dx * 0.1, Math.min(dy, 0) * 0.2 - 14, -3, 1, 0), offset: 0.14, easing: 'cubic-bezier(.4, 0, .2, 1)' },
      { ...pose(dx, dy, 0, s, 0), offset: 0.48, easing: 'cubic-bezier(.45, 0, .55, 1)' },
      { ...pose(dx, dy + h * s, 0, s, 100), offset: 1 },
    ], {
      duration: kind === 'insert' ? HOLD.insert : HOLD.eject,
      direction: kind === 'insert' ? 'normal' : 'reverse',
      fill: 'forwards',
    });
  }

  function paintScore(result) {
    const p = result.prediction;
    const t = p.timings_seconds ?? {};
    const v = p.verified && typeof p.verified === 'object' ? p.verified : null;
    $('.osd-score-value').textContent = scoreText(p.score);
    root.dataset.range = p.score < 1 || p.score > 5 ? 'outside' : 'inside';
    const meta = [];
    const total = secondsText(t.total);
    if (total) meta.push(`ANALYSIS ${total.toUpperCase()}`);
    if (typeof v?.model_sha256 === 'string') meta.push(`MODEL ${v.model_sha256.slice(0, 8).toUpperCase()}`);
    if (v) meta.push('VERIFIED');
    $('.osd-meta').textContent = meta.join('   ·   ');
  }

  function draw(name, s) {
    view = name;
    root.dataset.view = name;
    root.dataset.phase = phase;
    $('.osd-tl').textContent = CORNER[name] ?? '';
    $('.osd-main-text').textContent = name === 'processing' ? STAGE[s.stage] ?? 'PROCESSING' : MAIN[name] ?? '';
    $('.osd-sub').textContent = name === 'standby' ? 'MODEL OFFLINE'
      : name === 'invalid' || name === 'error' ? REASON[s.error?.code] ?? 'SCORING FAILED' : '';
    const filled = Math.round((s.uploadPct / 100) * CELLS);
    [...cells.children].forEach((c, i) => { c.className = i < filled ? 'on' : ''; });
    $('.osd-pct').textContent = `${s.uploadPct}%`;
    if (name === 'done' && s.result) paintScore(s.result);
    tape.disabled = INSIDE.has(name);
    eject.disabled = !(name === 'done' || name === 'error');
    paintClocks();
    syncNoise();
    syncClock();
    // The menu takes focus once, when the score screen is actually showing.
    if (name === 'done' && phase !== 'burst' && s.result && focusedFor !== s.result) {
      focusedFor = s.result;
      pick(0);
    }
    if (name !== 'done') focusedFor = null;
  }

  function apply(s) {
    const name = s.name;
    const reduce = motion.matches;
    const entering = view !== name;

    // A new tape while the old one is inside: eject first, then insert on the next pass.
    if (name === 'inserting' && tapePos === 'inside' && s.file !== tapeFile && !reduce) {
      setTape('ejecting');
      playTape('eject');
      return hold(HOLD.eject);
    }

    const wantInside = INSIDE.has(name);
    if (wantInside && tapePos !== 'inside') {
      tapeFile = s.file;
      paintLabel(s.file);
      if (!reduce && tapePos !== 'inserting') {
        setTape('inserting');
        playTape('insert');
        draw('inserting', s); // the insert screen shows only for this one hold
        return hold(HOLD.insert);
      }
    }

    if (wantInside) {
      setTape('inside');
    } else if (tapePos === 'inside' || tapePos === 'inserting') {
      paintLabel(s.file);
      draw(name, s);
      if (reduce) return setTape('rest');
      setTape('ejecting');
      playTape('eject');
      return hold(HOLD.eject);
    } else {
      setTape('rest');
      paintLabel(s.file);
    }

    if (entering && name === 'done' && !reduce) {
      phase = 'burst';
      hold(HOLD.burst);
    }
    draw(name, s);
    // A refused tape shakes: a wrong file, or any tape offered while the model is offline.
    const refused = name === 'invalid' || (name === 'error' && s.error?.code === 'offline');
    if (entering && refused && !reduce) {
      tape.classList.remove('is-shaking');
      void tape.offsetWidth; // restart the animation on repeated rejections
      tape.classList.add('is-shaking');
      hold(HOLD.shake);
    }
  }

  const onVisibility = () => syncNoise();
  const onMotion = () => { if (latest) apply(latest); };
  document.addEventListener('visibilitychange', onVisibility);
  motion.addEventListener('change', onMotion);

  return {
    render(state) {
      latest = state;
      if (performance.now() < holdUntil) return; // the hold timer renders the latest state when it ends
      apply(state);
    },
    setExpanded(tab) {
      items[0].setAttribute('aria-expanded', String(tab === 'summary'));
      items[1].setAttribute('aria-expanded', String(tab === 'json'));
    },
    focusTape() {
      tape.focus();
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      clearTimeout(holdTimer);
      clearInterval(clockTimer);
      document.removeEventListener('visibilitychange', onVisibility);
      motion.removeEventListener('change', onMotion);
    },
  };
}
