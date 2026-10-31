// Hand-rolled JSON highlighter. Every value goes in through textContent, never innerHTML.

const TOKEN = /("(?:\\u[0-9a-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;

export const serialize = (value) => JSON.stringify(value, null, 2);

function kind(token, isKey) {
  if (isKey) return 'json-key';
  if (token[0] === '"') return 'json-string';
  if (token === 'true' || token === 'false' || token === 'null') return 'json-literal';
  return 'json-number';
}

export function renderJson(value) {
  const text = serialize(value);
  const pre = document.createElement('pre');
  pre.className = 'json-view';
  pre.tabIndex = 0; // scrollable region must be keyboard reachable
  pre.setAttribute('aria-label', 'Result JSON');
  const code = document.createElement('code');
  let last = 0;
  for (const match of text.matchAll(TOKEN)) {
    if (match.index > last) code.append(text.slice(last, match.index));
    const isKey = Boolean(match[2]);
    const span = document.createElement('span');
    span.className = kind(match[0], isKey);
    span.textContent = isKey ? match[0].slice(0, match[0].length - match[2].length) : match[0];
    code.append(span);
    if (isKey) code.append(match[2]);
    last = match.index + match[0].length;
  }
  code.append(text.slice(last));
  pre.append(code);
  return pre;
}

export function copyText(text) {
  if (!navigator.clipboard?.writeText) return Promise.reject(new Error('Clipboard unavailable'));
  return navigator.clipboard.writeText(text);
}

export function downloadJson(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
