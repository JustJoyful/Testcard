// Display-only formatting. Raw values are never rounded anywhere else.

export function scoreText(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError('Score is not a finite number');
  }
  return value.toFixed(2);
}

// Returns '' for missing or non-finite values so callers can omit the row.
export function secondsText(value, { head = false } = {}) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '';
  if (head && value < 0.01) return '< 0.01 s';
  return `${value.toFixed(2)} s`;
}

export function shortHash(hash) {
  return typeof hash === 'string' && hash.length > 8 ? `${hash.slice(0, 8)}…` : String(hash ?? '');
}

// Keeps the start and the extension visible: "Zeke Scre…t Aot.mp4".
export function middleEllipsis(text, max = 22) {
  const chars = Array.from(String(text ?? ''));
  if (chars.length <= max) return chars.join('');
  const tail = Math.floor((max - 1) / 2);
  return chars.slice(0, max - 1 - tail).join('') + '…' + chars.slice(-tail).join('');
}

// Elapsed wall-clock time for the VCR counter, e.g. "00:00:07".
export function clockText(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(Math.min(99, Math.floor(total / 3600)))}:${pad(Math.floor(total / 60) % 60)}:${pad(total % 60)}`;
}
