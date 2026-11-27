"""Draw the README figures as SVG, one light and one dark file each, from results/summary.json and
the per-epoch run records.

  python results/summarize.py && python figures/make_figures.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W = 760
FONT = 'system-ui, -apple-system, &quot;Segoe UI&quot;, Helvetica, Arial, sans-serif'
THEMES = {  # page = GitHub's background
    'light': dict(page='#ffffff', panel='#f6f5f1', ink='#0b0b0b', ink2='#52514e', muted='#8f8d86', grid='#e1e0d9', axis='#c3c2b7',
                  ours='#2a78d6', base='#1baf7a', paper='#eb6834', alt='#8a5cd0', soft='#cde2fb'),
    'dark': dict(page='#0d1117', panel='#161b22', ink='#f0efec', ink2='#c3c2b7', muted='#7d7b75', grid='#2c2c2a', axis='#4a4a46',
                 ours='#3987e5', base='#199e70', paper='#d95926', alt='#a07be0', soft='#1f3a5c'),
}
S = json.load(open(os.path.join(ROOT, 'results', 'summary.json')))


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Svg:
    def __init__(self, h, t, w=W):
        self.h, self.w, self.t, self.parts = h, w, t, []

    def c(self, name):
        return self.t.get(name, name)

    def text(self, x, y, s, size=13, fill='ink', anchor='start', weight=400, italic=False):
        st = ' font-style="italic"' if italic else ''
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{self.c(fill)}" '
                          f'text-anchor="{anchor}" font-weight="{weight}"{st}>{esc(s)}</text>')

    def lines(self, x, y, rows, size=12, fill='ink2', gap=None, anchor='start', weight=400):
        for k, row in enumerate(rows):
            self.text(x, y + k * (gap or size + 4), row, size=size, fill=fill, anchor=anchor, weight=weight)

    def line(self, x1, y1, x2, y2, stroke='grid', width=1, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        a = ' marker-end="url(#ah)"' if arrow else ''
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{self.c(stroke)}" stroke-width="{width}" stroke-linecap="round"{d}{a}/>')

    def rect(self, x, y, w, h, fill='panel', stroke=None, rx=0, width=1, dash=None, opacity=None):
        s = f' stroke="{self.c(stroke)}" stroke-width="{width}"' if stroke else ''
        d = f' stroke-dasharray="{dash}"' if dash else ''
        o = f' opacity="{opacity}"' if opacity is not None else ''
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{self.c(fill)}"{s}{d}{o}/>')

    def bar(self, x, y, w, h, fill):  # square at the baseline, 4px rounded at the data end
        w = max(w, 0.1)
        r = min(4, w)
        self.parts.append(f'<path d="M{x:.1f} {y:.1f} H{x + w - r:.1f} a{r} {r} 0 0 1 {r} {r} V{y + h - r:.1f} '
                          f'a{r} {r} 0 0 1 -{r} {r} H{x:.1f} Z" fill="{self.c(fill)}"/>')

    def polyline(self, pts, stroke, width=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        p = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{self.c(stroke)}" stroke-width="{width}" '
                          f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def dot(self, x, y, fill, r=5):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{self.c(fill)}" '
                          f'stroke="{self.t["page"]}" stroke-width="2"/>')

    def diamond(self, x, y, fill, r=6):
        self.parts.append(f'<path d="M{x:.1f} {y - r:.1f} L{x + r:.1f} {y:.1f} L{x:.1f} {y + r:.1f} L{x - r:.1f} {y:.1f} Z" '
                          f'fill="{self.c(fill)}" stroke="{self.t["page"]}" stroke-width="2"/>')

    def box(self, x, y, w, h, rows, hi=False, size=12.5):
        self.rect(x, y, w, h, fill='soft' if hi else 'panel', stroke='ours' if hi else 'axis', rx=7, width=1.5 if hi else 1)
        n = len(rows)
        for k, row in enumerate(rows):
            self.text(x + w / 2, y + h / 2 + (k - (n - 1) / 2) * (size + 3) + size * 0.35, row, size=size if k == 0 else size - 1.5,
                      fill='ink' if k == 0 else 'ink2', anchor='middle', weight=600 if k == 0 else 400)

    def title(self, title, sub=None):
        self.text(0, 20, title, size=16, weight=600)
        if sub:
            self.text(0, 40, sub, size=12.5, fill='ink2')

    def save(self, name, theme):
        defs = (f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
                f'<path d="M1 1 L7 4 L1 7" fill="none" stroke="{self.t["muted"]}" stroke-width="1.4"/></marker></defs>')
        body = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-16 -8 {self.w + 32} {self.h + 16}" width="{self.w + 32}" '
                f'font-family="{FONT}" role="img">{defs}<rect x="-16" y="-8" width="{self.w + 32}" height="{self.h + 16}" '
                f'fill="{self.t["page"]}"/>' + ''.join(self.parts) + '</svg>')
        with open(os.path.join(HERE, f'{name}_{theme}.svg'), 'w', encoding='utf-8') as fh:
            fh.write(body)


def epochs(run):
    path = os.path.join(ROOT, 'results', 'runs', run, 'epochs.jsonl')
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


pct = lambda v, d=1: f'{100 * v:.{d}f}%'
signed = lambda v, d=1: f'{100 * v:+.{d}f}%'.replace('-', '−')

# ------------------------------------------------------------------------------------------------
def fig_ladder(theme):
    t = THEMES[theme]
    sg, gr = S['single'], S['groups']
    share = lambda mse: 1 - mse / S['variance']
    rows = []  # label, note, mse, kind, (lo, hi) range of mse or None
    if 'as_shipped' in sg:
        rows.append(('Released code, as shipped', f"{sg['as_shipped']['epochs_run']} epochs", sg['as_shipped']['test'], 'ours', None))
    if 'both_fixes_matched' in sg:
        rows.append(('Released code, two bugs fixed', f"same {sg['both_fixes_matched']['matched_epochs']} epochs", sg['both_fixes_matched']['test'], 'ours', None))
    if 'both_fixes' in sg:
        rows.append(('Released code, two bugs fixed', f"{sg['both_fixes']['epochs_run']} epochs", sg['both_fixes']['test'], 'ours', None))
    if 'full' in gr:
        g = gr['full']
        rows.append(("Paper's equations, re-implemented", f"mean of {g['n']} seed{'s' if g['n'] > 1 else ''}", g['test_mean'], 'ours', (g['test_min'], g['test_max'])))
    if 'equations_bias' in sg:
        rows.append(("Paper's equations + user/item bias", '1 seed', sg['equations_bias']['test'], 'ours', None))
    if 'notarget' in gr:
        g = gr['notarget']
        rows.append(("Paper's equations, own review hidden", f"in training; mean of {g['n']} seed{'s' if g['n'] > 1 else ''}", g['test_mean'], 'ours', (g['test_min'], g['test_max'])))
    rows.append(('User + item biases, no text', 'the baseline', S['baselines']['bias']['test'], 'base', None))
    rows.append(('RPMIA as reported in the paper', 'not measured here', S['paper']['full'], 'paper', None))
    LW, PW, RH, top = 286, 380, 40, 78
    s = Svg(top + RH * len(rows) + 46, t)
    s.title('How much of the rating variance each model removes', 'Test set. 0% is predicting the average rating for everyone; higher is better.')
    lx = 0
    for kind, name in (('ours', 'RPMIA, our runs'), ('base', 'No-text baseline'), ('paper', 'Reported in the paper')):
        if kind == 'paper':
            s.rect(lx, 54, 16, 11, fill='page', stroke='paper', rx=2, width=1.6)
        else:
            s.rect(lx, 54, 16, 11, fill=kind, rx=2)
        s.text(lx + 22, 64, name, size=12, fill='ink2')
        lx += 44 + 7.2 * len(name)
    xmax = 0.40
    x = lambda v: LW + v / xmax * PW
    for tick in (0, 0.1, 0.2, 0.3, 0.4):
        s.line(x(tick), top - 4, x(tick), top + RH * len(rows), stroke='axis' if tick == 0 else 'grid')
        s.text(x(tick), top + RH * len(rows) + 16, pct(tick, 0), size=11.5, fill='muted', anchor='middle')
    for k, (label, note, mse, kind, rng) in enumerate(rows):
        y = top + k * RH
        s.text(0, y + 16, label, size=13, weight=600 if kind != 'ours' else 400)
        s.text(0, y + 31, note, size=11, fill='muted')
        v = share(mse)
        if kind == 'paper':
            s.rect(x(0), y + 9, x(v) - x(0), 18, fill='page', stroke='paper', rx=3, width=1.6, dash='5 3')
        else:
            s.bar(x(0), y + 9, x(v) - x(0), 18, kind)
        if rng and rng[0] != rng[1]:
            a, b = share(rng[1]), share(rng[0])
            s.line(x(a), y + 18, x(b), y + 18, stroke='ink', width=1.6)
            for e in (a, b):
                s.line(x(e), y + 13, x(e), y + 23, stroke='ink', width=1.6)
        end = x(max(v, share(rng[0]) if rng else v)) + 8
        s.text(end, y + 22, pct(v), size=13, weight=600)
        s.text(end + 50, y + 22, f'MSE {mse:.3f}', size=11, fill='muted')
    s.text(0, top + RH * len(rows) + 38, 'Whiskers: lowest and highest seed. Each run is scored at its best validation epoch.', size=11, fill='muted')
    s.save('ladder', theme)


# ------------------------------------------------------------------------------------------------
def fig_ablation(theme):
    gr = S['groups']
    if not all(k in gr for k in ('full', 'context', 'aspect', 'uiap')):
        return False
    t = THEMES[theme]
    mods = [('context', 'Context module removed'), ('aspect', 'Aspect module removed'), ('uiap', 'Aspect-pair module removed')]
    LW, PW, RH, top = 230, 470, 64, 92
    s = Svg(top + RH * len(mods) + 60, t)
    s.title('What removing each module does to the error', 'Change in test MSE against the full model. Right of zero means the module was helping.')
    s.diamond(6, 60, 'paper'); s.text(18, 64, 'reported in the paper (Table 7)', size=12, fill='ink2')
    s.dot(236, 60, 'ours'); s.text(248, 64, 'our runs: mean and range of 3 seeds', size=12, fill='ink2')
    s.rect(486, 54, 16, 11, fill='soft', rx=2); s.text(508, 64, 'full model, seed-to-seed spread', size=12, fill='ink2')
    vals = [v for m, _ in mods for v in gr[m]['rel_each']] + [S['paper'][m + '_rel'] for m, _ in mods] + gr['full']['rel_each']
    lo, hi = min(min(vals), -0.02), max(max(vals), 0.07)
    lo, hi = (int(lo * 100) - 1) / 100, (int(hi * 100) + 1) / 100
    x = lambda v: LW + (v - lo) / (hi - lo) * PW
    fl, fh = min(gr['full']['rel_each']), max(gr['full']['rel_each'])
    s.rect(x(fl), top - 6, x(fh) - x(fl), RH * len(mods) + 6, fill='soft', opacity=0.9)
    step = 0.02
    tick = int(lo / step) * step
    while tick <= hi + 1e-9:
        s.line(x(tick), top - 6, x(tick), top + RH * len(mods), stroke='axis' if abs(tick) < 1e-9 else 'grid', width=1.5 if abs(tick) < 1e-9 else 1)
        s.text(x(tick), top + RH * len(mods) + 16, signed(tick, 0), size=11.5, fill='muted', anchor='middle')
        tick += step
    for k, (m, label) in enumerate(mods):
        y = top + k * RH
        g = gr[m]
        s.text(0, y + 24, label, size=13, weight=600)
        s.text(0, y + 40, f"MSE {g['test_mean']:.3f} vs {gr['full']['test_mean']:.3f}", size=11, fill='muted')
        py = y + 18
        s.diamond(x(S['paper'][m + '_rel']), py, 'paper')
        s.text(x(S['paper'][m + '_rel']) + 11, py + 4, signed(S['paper'][m + '_rel']), size=12, fill='ink2')
        oy = y + 42
        s.line(x(min(g['rel_each'])), oy, x(max(g['rel_each'])), oy, stroke='ours', width=3)
        s.dot(x(g['rel_mean']), oy, 'ours')
        s.text(x(max(g['rel_each'])) + 11, oy + 4, signed(g['rel_mean']), size=12, weight=600)
    s.text(0, top + RH * len(mods) + 42, 'Our "removed" variants are defined in code/RPMIA_paper.py; the paper does not say what replaces a removed module.', size=11, fill='muted')
    s.save('ablation', theme)
    return True


# ------------------------------------------------------------------------------------------------
def fig_curves(theme):
    t = THEMES[theme]
    a, b = epochs('eq_s1')[:40], epochs('notarget_s1')
    if not a:
        return False
    L, R, top, H = 46, 150, 78, 250
    s = Svg(top + H + 44, t)
    s.title('The model memorises the training set', "Paper's equations, seed 1. Error on training pairs keeps falling; error on unseen pairs does not.")
    n = max(len(a), len(b))
    vals = [r['val_mse'] for r in a + b] + [r['train_mse'] for r in a + b]
    lo, hi = 0.5, min(1.5, max(vals) + 0.05)
    x = lambda e: L + e / max(n - 1, 1) * (W - L - R)
    y = lambda v: top + (hi - min(max(v, lo), hi)) / (hi - lo) * H
    tick = lo
    while tick <= hi + 1e-9:
        s.line(L, y(tick), W - R, y(tick), stroke='grid')
        s.text(L - 8, y(tick) + 4, f'{tick:.1f}', size=11.5, fill='muted', anchor='end')
        tick += 0.2
    for e in range(0, n, 5):
        s.text(x(e), top + H + 16, str(e + 1), size=11.5, fill='muted', anchor='middle')
    s.text((L + W - R) / 2, top + H + 34, 'training epoch', size=11.5, fill='muted', anchor='middle')
    s.text(L - 8, top - 10, 'MSE', size=11.5, fill='muted', anchor='end')
    bias = S['baselines']['bias']['val']
    s.line(L, y(bias), W - R, y(bias), stroke='base', width=1.5, dash='5 4')
    s.text(W - R + 8, y(bias) + 4, f'bias baseline {bias:.2f}', size=11.5, fill='ink2')
    series = [(a, 'ours', 'standard')] + ([(b, 'alt', 'own review hidden')] if b else [])
    for rows, colour, name in series:
        s.polyline([(x(k), y(r['train_mse'])) for k, r in enumerate(rows)], colour, width=2, dash='2 5')
        s.polyline([(x(k), y(r['val_mse'])) for k, r in enumerate(rows)], colour, width=2.2)
        best = min(range(len(rows)), key=lambda k: rows[k]['val_mse'])
        s.dot(x(best), y(rows[best]['val_mse']), colour)
        last = rows[-1]
        tag = '' if len(series) == 1 else f' ({name})'
        s.text(x(len(rows) - 1) + 8, y(last['val_mse']) + 4, 'unseen pairs' + tag, size=11.5, fill='ink2')
        s.text(x(len(rows) - 1) + 8, y(last['train_mse']) + 4, 'training pairs' + tag, size=11.5, fill='ink2')
    s.save('curves', theme)
    return True


# ------------------------------------------------------------------------------------------------
def fig_ratings(theme):
    t = THEMES[theme]
    counts = json.load(open(os.path.join(ROOT, 'results', 'rating_counts.json')))
    total = sum(counts.values())
    L, top, H, BW = 46, 62, 190, 24
    s = Svg(top + H + 60, t, w=W)
    s.title('Why two numbers per pair already explain a lot', 'Share of all ratings at each star level. Most of what varies is how generous a user is and how good an item is.')
    x = lambda k: L + 60 + k * 110
    ymax = 0.6
    y = lambda v: top + (1 - v / ymax) * H
    for tick in (0, 0.2, 0.4, 0.6):
        s.line(L, y(tick), L + 590, y(tick), stroke='axis' if tick == 0 else 'grid')
        s.text(L - 8, y(tick) + 4, pct(tick, 0), size=11.5, fill='muted', anchor='end')
    for k, star in enumerate(('1', '2', '3', '4', '5')):
        v = counts[star] / total
        h = y(0) - y(v)
        r = 4
        s.parts.append(f'<path d="M{x(k) - BW / 2:.1f} {y(0):.1f} V{y(v) + r:.1f} a{r} {r} 0 0 1 {r} -{r} H{x(k) + BW / 2 - r:.1f} '
                       f'a{r} {r} 0 0 1 {r} {r} V{y(0):.1f} Z" fill="{t["base"]}"/>')
        s.text(x(k), y(v) - 8, pct(v), size=13, weight=600, anchor='middle')
        s.text(x(k), y(0) + 18, f'{star} star' + ('' if star == '1' else 's'), size=12, fill='ink2', anchor='middle')
    s.text(L, top + H + 46, f'{total:,} ratings from 14,681 users on 8,713 items; about 7 training ratings per user.', size=11, fill='muted')
    s.save('ratings', theme)


# ------------------------------------------------------------------------------------------------
def fig_architecture(theme):
    t = THEMES[theme]
    s = Svg(668, t)
    s.title('RPMIA, module by module', 'Redrawn after Figure 1 of the paper. Right column: what the paper says each module is for.')
    LU, LI, BW = 0, 196, 180      # user lane x, item lane x, box width
    AX = 410                      # annotation column
    span = LI + BW - LU

    def arrow(x, y1, y2):
        s.line(x, y1, x, y2 - 3, stroke='muted', width=1.4, arrow=True)

    def note(y, head, eq, rows):
        s.text(AX, y + 14, head, size=13.5, weight=600)
        s.text(AX, y + 30, eq, size=11, fill='muted')
        s.lines(AX, y + 48, rows, size=12, fill='ink2', gap=16)

    y = 62
    s.box(LU, y, BW, 44, ['User document', 'every review the user wrote']); s.box(LI, y, BW, 44, ['Item document', 'every review the item got'])
    note(y - 2, 'Inputs', 'up to 300 words each', ['The same pipeline runs on both sides.'])
    for lx in (LU, LI):
        arrow(lx + BW / 2, y + 44, y + 70)
    y = 132
    s.box(LU, y, span, 36, ['Word embedding'])
    note(y - 4, 'Embedding', 'Eq. 1', ['Pretrained Word2Vec vectors, 50-d.'])
    for lx in (LU, LI):
        arrow(lx + BW / 2, y + 36, y + 62)
    y = 194
    s.rect(LU - 8, y - 8, span + 16, 128, fill='page', stroke='ours', rx=10, width=1.2, dash='4 4')
    s.box(LU, y, BW, 40, ['Convolutions', 'a local context per word']); s.box(LI, y, BW, 40, ['Convolutions', 'a local context per word'])
    for lx in (LU, LI):
        arrow(lx + BW / 2, y + 40, y + 66)
    s.box(LU, y + 66, span, 46, ['Multi-head cross-attention', "each side's words weighted against the other side"], hi=True)
    note(y - 4, 'Context-aware interaction', 'Eq. 2–7 · level 1',
         ['Stated purpose: decide which words matter', 'for this user–item pair, from several', 'views at once.'])
    for lx in (LU, LI):
        arrow(lx + BW / 2, y + 112, y + 146)
    y = 340
    s.rect(LU - 8, y - 8, span + 16, 122, fill='page', stroke='ours', rx=10, width=1.2, dash='4 4')
    s.box(LU, y, BW, 40, ['Two-layer gate', '5 learned aspect vectors'], hi=True); s.box(LI, y, BW, 40, ['Two-layer gate', '5 learned aspect vectors'], hi=True)
    for lx in (LU, LI):
        arrow(lx + BW / 2, y + 40, y + 66)
    s.box(LU, y + 66, BW, 40, ['Self-attention', '5 user aspects'], hi=True); s.box(LI, y + 66, BW, 40, ['Self-attention', '5 item aspects'], hi=True)
    note(y - 4, 'Aspect-aware interaction', 'Eq. 8–10 · level 2',
         ['Stated purpose: distil the aspects each', 'side cares about and weight them by', 'importance.'])
    for lx in (LU, LI):
        arrow(lx + BW / 2, y + 106, y + 140)
    y = 480
    s.rect(LU - 8, y - 8, span + 16, 62, fill='page', stroke='ours', rx=10, width=1.2, dash='4 4')
    s.box(LU, y, BW, 46, ['5 × 5 aspect pairs', 'each user aspect × each item aspect'], hi=True, size=12)
    s.box(LI, y, BW, 46, ['Joint attention', 'weights the 25 pairs'], hi=True)
    s.line(LU + BW + 2, y + 23, LI - 4, y + 23, stroke='muted', width=1.4, arrow=True)
    note(y - 6, 'Aspect-pair (UIAP) interaction', 'Eq. 11–13 · level 3',
         ['Stated purpose: capture the joint effect of', 'a preference and the matching item trait.'])
    arrow(LI + BW / 2, y + 46, y + 78)
    y = 558
    s.box(LU, y, 84, 40, ['FM']); s.box(LU + 96, y, 84, 40, ['FNN'])
    s.box(LI, y, BW, 40, ['Sum, then sigmoid', 'predicted rating'])
    s.line(LU + BW + 2, y + 20, LI - 4, y + 20, stroke='muted', width=1.4, arrow=True)
    s.line(LI + BW / 2, y, LI + BW / 2, y - 8, stroke='muted', width=1.4)
    s.line(LI + BW / 2, y - 8, LU + 42, y - 8, stroke='muted', width=1.4)
    s.line(LU + 42, y - 8, LU + 42, y - 3, stroke='muted', width=1.4, arrow=True)
    s.line(LU + 138, y - 8, LU + 138, y - 3, stroke='muted', width=1.4, arrow=True)
    note(y - 6, 'Prediction', 'Eq. 14–16',
         ['Stated purpose: a factorisation machine for', 'the direct effect of the pairs, a feed-forward', 'network for deeper associations.'])
    s.text(0, 640, 'Dashed frames: the three interaction levels the paper presents as its contribution.', size=11, fill='muted')
    s.save('architecture', theme)


# ------------------------------------------------------------------------------------------------
def fig_broadcast(theme):
    t = THEMES[theme]
    s = Svg(300, t)
    s.title('The shape bug: four predictions become sixteen', 'Drawn for a batch of 4. One side has shape [B, 1], the other [B]; subtracting them gives [B, B].')
    cell, x0, y0 = 30, 6, 96

    def grid(x, y, rows, cols, fill_fn, stroke='axis'):
        for r in range(rows):
            for c in range(cols):
                s.rect(x + c * cell, y + r * cell, cell - 3, cell - 3, fill=fill_fn(r, c), stroke=stroke, rx=4)

    s.text(x0 + 13, y0 - 12, 'J₂', size=13, weight=600, anchor='middle')
    grid(x0, y0, 4, 1, lambda r, c: 'panel')
    s.text(x0 + 52, y0 + 66, '−', size=20, fill='ink2', anchor='middle')
    s.text(x0 + 78 + 58, y0 + 32, 'J₃', size=13, weight=600, anchor='middle')
    grid(x0 + 78, y0 + 45, 1, 4, lambda r, c: 'panel')
    s.text(x0 + 222, y0 + 66, '=', size=20, fill='ink2', anchor='middle')
    gx = x0 + 250
    grid(gx, y0, 4, 4, lambda r, c: 'ours' if r == c else 'panel')
    s.rect(gx - 4, y0 - 4, cell + 4, 4 * cell + 4, fill='none', stroke='paper', rx=6, width=2)
    s.text(gx + 60, y0 - 12, 'as shipped: [B, B]', size=12.5, weight=600, anchor='middle')
    lx = gx + 4 * cell + 22
    s.rect(lx, y0 + 6, 14, 14, fill='ours', rx=3); s.lines(lx + 22, y0 + 18, ['the correct values', 'sit on the diagonal'], size=12, gap=15)
    s.rect(lx, y0 + 52, 14, 14, fill='none', stroke='paper', rx=3, width=2); s.lines(lx + 22, y0 + 64, ['evaluation reads', 'column 0 only'], size=12, gap=15)
    s.lines(lx, y0 + 108, ['training compares each rating', 'with its whole row'], size=12, gap=15)
    fx = 620
    s.text(fx + 13, y0 - 12, 'fixed: [B, 1]', size=12.5, weight=600, anchor='middle')
    grid(fx, y0, 4, 1, lambda r, c: 'ours')
    s.text(0, 262, 'The fix is one index: J_total = J_1 + 0.5 * (J_2 - J_3[:, None]).', size=12, fill='ink2')
    s.text(0, 282, 'A prediction then depends only on its own pair, not on what else is in the batch.', size=12, fill='ink2')
    s.save('broadcast', theme)


# ------------------------------------------------------------------------------------------------
def fig_rebuild(theme):
    t = THEMES[theme]
    s = Svg(300, t)
    s.title('Rebuilding the inputs the repository leaves out', 'The loader needs every user\'s and item\'s review text; the repository ships ratings, ids and a dictionary only.')
    steps = [['Raw reviews', 'Amazon Grocery,', '2014 5-core file'], ['Recover user ids', 'join on item, rating', 'and timestamp'],
             ['Tokenise', 'lowercase, keep', 'dictionary words'], ['Build documents', 'training-split', 'reviews only']]
    bw, gap, y = 166, 32, 66
    for k, rows in enumerate(steps):
        x = k * (bw + gap)
        s.box(x, y, bw, 70, rows)
        if k:
            s.line(x - gap + 3, y + 35, x - 4, y + 35, stroke='muted', width=1.4, arrow=True)
    checks = [('151,226 of 151,226', 'rating rows matched to one reviewer'),
              ('19,987 of 19,987', "dictionary words found in the 70% split"),
              ('0', 'validation or test reviews in any document')]
    cw = (W - 2 * 16) / 3
    for k, (big, small) in enumerate(checks):
        x = k * (cw + 16)
        s.rect(x, 168, cw, 84, fill='panel', stroke='axis', rx=8)
        s.text(x + 14, 204, big, size=21, weight=700)
        s.text(x + 14, 228, small, size=11.5, fill='ink2')
    s.text(0, 280, 'Checks on the rebuild. The second one also shows which split the authors built their documents from.', size=11, fill='muted')
    s.save('rebuild', theme)


if __name__ == '__main__':
    made = []
    for theme in THEMES:
        fig_ladder(theme); fig_ratings(theme); fig_architecture(theme); fig_broadcast(theme); fig_rebuild(theme)
        ok_a, ok_c = fig_ablation(theme), fig_curves(theme)
    made = ['ladder', 'ratings', 'architecture', 'broadcast', 'rebuild'] + (['ablation'] if ok_a else []) + (['curves'] if ok_c else [])
    print('figures:', ', '.join(made))
