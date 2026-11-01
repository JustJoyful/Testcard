"""Frontend acceptance against the running Phase 4b image; writes results/phase4/frontend_acceptance.{json,md}.

    .venv-serving/bin/python -m results.phase4.frontend.acceptance

Needs the vqa:phase4b container on 127.0.0.1:8001 (name vqa-phase4b-check) and serving/web/node_modules.
Every pass/fail below is derived from outputs saved in results/phase4/frontend/.
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

from src.cache import atomic_json, atomic_write

OUT = Path('results/phase4/frontend')
WEB = Path('serving/web')
URL, IMAGE, CONTAINER = 'http://127.0.0.1:8001', 'vqa:phase4b', 'vqa-phase4b-check'
VIDEO = Path('data/raw/konvid') / (Path('data/splits/phase0_konvid_ids.txt').read_text().split()[0] + '.mp4')
VERIFIED = ('model_sha256', 'weights_sha256', 'preprocessing_sha256')


def run(command, env=None, cwd=None, log=None):
    result = subprocess.run(command, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True)
    if log:
        atomic_write(OUT / log, (result.stdout + result.stderr).encode())
    return result


def playwright(name, env):
    """Run one Playwright configuration with the JSON reporter; return {test title: status}."""
    report = (OUT / f'{name}.report.json').resolve()
    result = run(['npx', 'playwright', 'test', '--reporter=json'], cwd=WEB, log=f'{name}.log',
                 env={**env, 'PLAYWRIGHT_JSON_OUTPUT_NAME': str(report)})
    statuses = {}
    def walk(suite):
        for spec in suite.get('specs', []):
            for test in spec['tests']:
                statuses[spec['title']] = test['results'][-1]['status'] if test['results'] else test['status']
        for child in suite.get('suites', []):
            walk(child)
    for suite in json.loads(report.read_text())['suites']:
        walk(suite)
    return dict(exit_code=result.returncode, tests=statuses)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    def item(name, passed, how, evidence):
        items.append(dict(item=name, status='pass' if passed else 'fail', how=how, evidence=evidence))

    image_id = run(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']).stdout.strip()
    if run(['docker', 'inspect', CONTAINER, '--format', '{{.Image}}']).stdout.strip() != image_id:
        raise SystemExit('the running container is not the current vqa:phase4b image; restart it first')
    health = httpx.get(URL + '/health', timeout=10).json()

    # Real-backend browser checks: keyboard flow, what the page shows, offline requests, 360 px.
    container = playwright('container', {'VQA_CONTAINER_URL': URL, 'VQA_REAL_VIDEO': str(VIDEO.resolve()),
                                         'VQA_ACCEPTANCE_OUT': str(OUT.resolve())})
    ui = json.loads((OUT / 'ui_result.json').read_text())
    prediction = ui['json']['prediction']

    # Hashes: page == CLI --json for the same file in the same image == the running API's startup log.
    logs = run(['docker', 'logs', CONTAINER])
    line = next((l for l in (logs.stdout + logs.stderr).splitlines() if 'verified runtime: ' in l), None)
    startup = json.loads(line.split('verified runtime: ', 1)[1]) if line else None
    cli = run(['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}', '--entrypoint', 'python',
               '--mount', f'type=bind,src={VIDEO.resolve()},dst=/input/video.mp4,readonly', IMAGE, '-m', 'src.predict',
               '/input/video.mp4', '--json'], log='cli.log')
    cli_verified = json.loads(cli.stdout)['verified'] if cli.returncode == 0 else None
    page = prediction['verified']
    shown = {row['full']: row['text'] for row in ui['hashRows']}
    hashes_ok = (startup is not None and cli_verified is not None
                 and all(page[k] == cli_verified[k] == startup[k] for k in VERIFIED)
                 and all(shown.get(page[k]) == page[k][:8] + '…' for k in VERIFIED)
                 and f"MODEL {page['model_sha256'][:8].upper()}" in ui['osdMeta'])
    atomic_json(OUT / 'hashes.json', dict(page=page, page_short=ui['hashRows'], osd_meta=ui['osdMeta'], cli=cli_verified,
                                          startup_log_line=line, health=health.get('verified')))
    item('Hashes shown equal `src.predict --json` and the API startup log', hashes_ok,
         'JSON tab, summary hash rows (short text and full title) and OSD prefix compared with CLI `--json` on the same clip '
         'in the same image and with the `verified runtime:` line in `docker logs`', 'frontend/hashes.json')

    score = prediction['score']
    probe = run(['node', '--input-type=module', '-e', "import { scoreText } from './src/format.js'; "
                 "console.log(JSON.stringify([scoreText(5.256), scoreText(-0.254), scoreText(0.999)]))"], cwd=WEB)
    unclamped = probe.returncode == 0 and json.loads(probe.stdout) == ['5.26', '-0.25', '1.00']
    item('Score equals `prediction.score` rounded to two decimals, unclamped', ui['scoreText'] == f'{score:.2f}' and unclamped,
         f"on-screen `{ui['scoreText']}` vs `{score!r}` in the JSON tab; `scoreText(5.256)`, `scoreText(-0.254)` stay outside 1–5",
         'frontend/ui_result.json')
    keyboard = 'keyboard only on the real backend: insert, result, details, JSON, eject'
    item('Full flow idle → result → details → JSON → eject by keyboard only', container['tests'].get(keyboard) == 'passed',
         'Playwright on the running image: Tab to the tape, Enter, real upload, Enter / ArrowRight / End, focus back on the tape',
         'frontend/container.log')
    offline = ('built page and every font load from the image alone', 'a real clip is scored end to end offline')
    item('Built page makes no requests to external domains', all(container['tests'].get(t) == 'passed' for t in offline),
         'every non-origin request aborted and recorded; all fonts loaded and a real clip scored with zero blocked requests',
         'frontend/container.log')
    item('No horizontal scroll at 360 px', container['tests'].get('360 px: no horizontal scroll at idle or with a real result') == 'passed',
         '360×780 touch viewport: `scrollWidth` ≤ 360 at idle and after a real result with the JSON tab open', 'frontend/container.log')

    # Backend stopped: preview the built dist/ with its API proxy pointed at a closed port.
    run(['npx', 'vite', 'build'], cwd=WEB, log='vite_build.log')
    preview = subprocess.Popen(['npx', 'vite', 'preview', '--port', '4180', '--strictPort'], cwd=WEB, start_new_session=True,
                               env={**os.environ, 'VQA_API_URL': 'http://127.0.0.1:9'},
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                httpx.get('http://127.0.0.1:4180/', timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(.5)
        standby = playwright('standby', {'VQA_STANDBY_URL': 'http://127.0.0.1:4180'})
    finally:
        os.killpg(preview.pid, signal.SIGTERM)
    item('Backend stopped: page shows standby and never uploads',
         standby['tests'].get('backend stopped: the TV shows standby and nothing is ever uploaded') == 'passed',
         'built dist/ previewed with its API proxy pointed at a closed port; every request to `/upload` recorded (none)',
         'frontend/standby.log')

    # Mock mode in production: search every shipped file inside the image.
    grep = run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'grep', IMAGE, '-rlE',
                'analyzeMock|23\\.884803736000322|mock\\.js', '/app/serving/web/dist'], log='mock_grep.log')
    item('Mock mode cannot activate in a production build',
         grep.returncode == 1 and container['tests'].get('production ?mock still uses the real backend') == 'passed',
         'grep of /app/serving/web/dist in the image for the mock transport, its fixture timing and module name (no match), '
         'plus `/?mock` on the image keeps the badge hidden and uses the real backend', 'frontend/mock_grep.log')

    chrome = sorted(Path.home().glob('.cache/ms-playwright/chromium-*/chrome-linux*/chrome'))[-1]
    lighthouse = (OUT / 'lighthouse.json').resolve()
    run(['npx', 'lighthouse', URL + '/', '--only-categories=accessibility', '--preset=desktop', '--output=json',
         f'--output-path={lighthouse}', '--chrome-flags=--headless=new', '--quiet'], cwd=WEB, env={'CHROME_PATH': str(chrome)})
    a11y = json.loads(lighthouse.read_text())['categories']['accessibility']['score']
    item('Lighthouse accessibility ≥ 95', a11y is not None and a11y >= .95,
         f'Lighthouse accessibility on the running image: {a11y * 100:.0f}', 'frontend/lighthouse.json')

    # Remaining behaviours (states, errors and retry, reduced motion, hidden-tab noise, JSON copy/download, phone).
    suite = playwright('demo', {})
    item('Full browser suite: states, errors and retry, reduced motion, noise pauses when hidden, JSON copy/download, phone',
         suite['exit_code'] == 0 and all(s in ('passed', 'skipped') for s in suite['tests'].values()),
         f"{sum(s == 'passed' for s in suite['tests'].values())} passed, {sum(s == 'skipped' for s in suite['tests'].values())} "
         'skipped (the dev-server real-backend test, covered above on the image)', 'frontend/demo.log')

    record = dict(image_id=image_id, clip=VIDEO.name, items=items, passed=all(i['status'] == 'pass' for i in items),
                  container_tests=container['tests'], standby_tests=standby['tests'], demo_tests=suite['tests'])
    atomic_json(Path('results/phase4/frontend_acceptance.json'), record)
    lines = ['# Frontend acceptance', '', f'Generated by `results/phase4/frontend/acceptance.py` against image `{image_id}` '
             f'on {URL}, real clip `{VIDEO.name}`. Evidence paths are relative to `results/phase4/`.', '',
             '| # | Item | Result | How it was checked | Evidence |', '|---|---|---|---|---|']
    lines += [f"| {n} | {i['item']} | **{i['status']}** | {i['how']} | `{i['evidence']}` |" for n, i in enumerate(items, 1)]
    lines += ['', f"Overall: **{'all items pass' if record['passed'] else 'failures listed above'}**.", '']
    atomic_write(Path('results/phase4/frontend_acceptance.md'), '\n'.join(lines).encode())
    print(json.dumps({i['item']: i['status'] for i in items}, indent=1))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
