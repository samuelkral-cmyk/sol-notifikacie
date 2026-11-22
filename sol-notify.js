/* SOL Forecast engine — pure JS, no DOM.
 * Volatility: intraday seasonality x GARCH(1,1), shocks bootstrapped jointly for SOL+BTC
 * (filtered historical simulation). Direction: L2 logistic model with SOL + BTC features,
 * walk-forward with purged labels and expanding Platt calibration (shrinks to 50 % when
 * there is no measured skill).
 */
(function (root) {
  'use strict';
  var E = {};
  var INTERVAL_MS = { '5m': 300000, '15m': 900000, '1h': 3600000, '4h': 14400000 };
  E.INTERVAL_MS = INTERVAL_MS;

  // ---------------------------------------------------------------- utilities
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeGauss(rnd) {
    var spare = null;
    return function () {
      if (spare !== null) { var s0 = spare; spare = null; return s0; }
      var u, v, s;
      do { u = rnd() * 2 - 1; v = rnd() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
      var m = Math.sqrt(-2 * Math.log(s) / s);
      spare = v * m;
      return u * m;
    };
  }
  E.mulberry32 = mulberry32;

  function qSorted(a, q) {
    var n = a.length;
    if (!n) return NaN;
    var pos = (n - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return a[lo] + (a[hi] - a[lo]) * (pos - lo);
  }
  function quantile(arr, q) { return qSorted(Float64Array.from(arr).sort(), q); }
  E.quantile = quantile;
  function sigmoid(z) { return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)); }
  function logit(p) { p = Math.min(1 - 1e-6, Math.max(1e-6, p)); return Math.log(p / (1 - p)); }
  function clip(x, a, b) { return x < a ? a : (x > b ? b : x); }

  function bucketOf(t) {
    var d = new Date(t), wd = d.getUTCDay();
    return d.getUTCHours() + ((wd === 0 || wd === 6) ? 24 : 0);
  }
  E.bucketOf = bucketOf;

  function logReturns(c) {
    var n = c.length, r = new Float64Array(n);
    for (var i = 1; i < n; i++) r[i] = Math.log(c[i] / c[i - 1]);
    return r;
  }

  // Solve A x = b for symmetric positive definite A (n x n, row-major). Returns x.
  function cholSolve(A, b, n) {
    var L = new Float64Array(n * n), i, j, k, s;
    for (i = 0; i < n; i++) {
      for (j = 0; j <= i; j++) {
        s = A[i * n + j];
        for (k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) L[i * n + i] = Math.sqrt(Math.max(s, 1e-12));
        else L[i * n + j] = s / L[j * n + j];
      }
    }
    var y = new Float64Array(n), x = new Float64Array(n);
    for (i = 0; i < n; i++) { s = b[i]; for (k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
    for (i = n - 1; i >= 0; i--) { s = y[i]; for (k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
    return x;
  }

  // ---------------------------------------------------------------- data prep
  // series: {t, o, h, l, c, v, tb} (arrays). Returns both aligned on open time.
  E.align = function (sol, btc) {
    var mb = new Map(), i;
    for (i = 0; i < btc.t.length; i++) mb.set(btc.t[i], i);
    var iS = [], iB = [];
    for (i = 0; i < sol.t.length; i++) {
      var j = mb.get(sol.t[i]);
      if (j !== undefined) { iS.push(i); iB.push(j); }
    }
    function pick(s, idx) {
      var out = {};
      ['t', 'o', 'h', 'l', 'c', 'v', 'tb'].forEach(function (k) {
        var src = s[k];
        out[k] = Float64Array.from(idx, function (q) { return src ? +src[q] : NaN; });
      });
      return out;
    }
    return { s: pick(sol, iS), b: pick(btc, iB), n: iS.length };
  };

  function fitSeasonality(r, bucket, a, b, intervalMs) {
    var sum = new Float64Array(48), cnt = new Float64Array(48), tot = 0, n = 0, i, k;
    for (i = Math.max(1, a); i < b; i++) {
      var v = Math.abs(r[i]);
      if (!isFinite(v)) continue;
      sum[bucket[i]] += v; cnt[bucket[i]]++; tot += v; n++;
    }
    var overall = tot / Math.max(1, n), prof = new Float64Array(48), K = 8;
    for (k = 0; k < 48; k++) {
      var m = cnt[k] ? (sum[k] / cnt[k]) / overall : 1;
      prof[k] = (cnt[k] * m + K) / (cnt[k] + K);
    }
    if (intervalMs <= 3600000) {
      var sm = new Float64Array(48);
      for (var half = 0; half < 2; half++) for (var hh = 0; hh < 24; hh++) {
        var c0 = half * 24 + hh;
        sm[c0] = 0.25 * prof[half * 24 + (hh + 23) % 24] + 0.5 * prof[c0] + 0.25 * prof[half * 24 + (hh + 1) % 24];
      }
      prof = sm;
    }
    var s2 = 0, w = 0;
    for (k = 0; k < 48; k++) { s2 += cnt[k] * prof[k] * prof[k]; w += cnt[k]; }
    var norm = Math.sqrt(s2 / Math.max(1, w)) || 1;
    for (k = 0; k < 48; k++) prof[k] /= norm;
    return prof;
  }

  function garchFit(x, a0, a1) {
    var n = a1 - a0, i, m = 0;
    for (i = a0; i < a1; i++) m += x[i];
    m /= n;
    var y = new Float64Array(n), v = 0;
    for (i = 0; i < n; i++) { y[i] = x[a0 + i] - m; v += y[i] * y[i]; }
    v /= n;
    function nll(a, b) {
      var w = v * (1 - a - b), h = v, L = 0;
      for (var q = 0; q < n; q++) { var yy = y[q] * y[q]; L += Math.log(h) + yy / h; h = w + a * yy + b * h; }
      return L;
    }
    var best = { a: 0.05, b: 0.9, L: Infinity }, a, b;
    for (a = 0.01; a <= 0.3001; a += 0.01) for (b = 0.5; b <= 0.986; b += 0.015) {
      if (a + b >= 0.998) continue;
      var L0 = nll(a, b);
      if (L0 < best.L) best = { a: a, b: b, L: L0 };
    }
    var sa = 0.005, sb = 0.0075;
    for (var it = 0; it < 3; it++) {
      var ca = best.a, cb = best.b;
      for (var da = -2; da <= 2; da++) for (var db = -2; db <= 2; db++) {
        var aa = ca + da * sa, bb = cb + db * sb;
        if (aa <= 0.002 || bb <= 0.3 || aa + bb >= 0.999) continue;
        var L1 = nll(aa, bb);
        if (L1 < best.L) best = { a: aa, b: bb, L: L1 };
      }
      sa /= 2; sb /= 2;
    }
    return { alpha: best.a, beta: best.b, omega: v * (1 - best.a - best.b), mu: m, uncVar: v };
  }

  // h[i] = conditional variance of bar i (info up to i-1); h[n] = next bar.
  function garchFilter(x, p) {
    var n = x.length, h = new Float64Array(n + 1);
    h[0] = p.uncVar; h[1] = p.uncVar;
    for (var i = 1; i < n; i++) { var y = x[i] - p.mu; h[i + 1] = p.omega + p.alpha * y * y + p.beta * h[i]; }
    return h;
  }

  // Volatility model for one symbol, parameters fitted on bars [1, fitEnd).
  function prepSym(S, intervalMs, fitEnd) {
    var n = S.c.length, i;
    var r = logReturns(S.c);
    var bucket = new Uint8Array(n);
    for (i = 0; i < n; i++) bucket[i] = bucketOf(S.t[i]);
    var season = fitSeasonality(r, bucket, 1, fitEnd, intervalMs);
    var x = new Float64Array(n);
    for (i = 1; i < n; i++) x[i] = r[i] / season[bucket[i]];
    var gp = garchFit(x, 1, fitEnd);
    var h = garchFilter(x, gp);
    var z = new Float64Array(n), U = new Float64Array(n), W = new Float64Array(n), sig = new Float64Array(n + 1);
    for (i = 1; i < n; i++) {
      z[i] = (x[i] - gp.mu) / Math.sqrt(h[i]);
      sig[i] = season[bucket[i]] * Math.sqrt(h[i]);
      var top = Math.max(S.o[i], S.c[i]), bot = Math.min(S.o[i], S.c[i]);
      var u = Math.log(S.h[i] / top) / sig[i], w = Math.log(bot / S.l[i]) / sig[i];
      U[i] = isFinite(u) && u > 0 ? Math.min(u, 8) : 0;
      W[i] = isFinite(w) && w > 0 ? Math.min(w, 8) : 0;
    }
    return { r: r, bucket: bucket, season: season, x: x, gp: gp, h: h, z: z, U: U, W: W, sig: sig };
  }

  // ---------------------------------------------------------------- chart patterns
  // Chart reading. Every pattern is detected causally: an event at candle i uses only candles <= i,
  // and never spans a gap in the saved history. Each has a textbook direction; how often it really
  // worked is measured on the same data (patternStats), so the advice can say "works here" or not.
  var PAT = {
    hammer:   { name: 'Kladivo', short: 'KL', dir: 1, kind: 'c', txt: 'dlhý dolný knôt po poklese' },
    star:     { name: 'Padajúca hviezda', short: 'PH', dir: -1, kind: 'c', txt: 'dlhý horný knôt po raste' },
    engBull:  { name: 'Býčie pohltenie', short: 'BP', dir: 1, kind: 'c', txt: 'zelená sviečka prekryla celú predošlú červenú' },
    engBear:  { name: 'Medvedie pohltenie', short: 'MP', dir: -1, kind: 'c', txt: 'červená sviečka prekryla celú predošlú zelenú' },
    mStar:    { name: 'Ranná hviezda', short: 'RH', dir: 1, kind: 'c', txt: 'po poklese malá sviečka a potom silná zelená' },
    eStar:    { name: 'Večerná hviezda', short: 'VH', dir: -1, kind: 'c', txt: 'po raste malá sviečka a potom silná červená' },
    soldiers: { name: 'Traja bieli vojaci', short: '3V', dir: 1, kind: 'c', txt: 'tri silné zelené sviečky za sebou' },
    crows:    { name: 'Tri čierne vrany', short: '3Č', dir: -1, kind: 'c', txt: 'tri silné červené sviečky za sebou' },
    doji:     { name: 'Doji', short: 'DJ', dir: 0, kind: 'c', txt: 'otvorenie takmer rovné zatvoreniu, nerozhodnosť' },
    dBot:     { name: 'Dvojité dno', short: 'DD', dir: 1, kind: 'g', txt: 'dve podobné dná a prieraz nad vrchol medzi nimi' },
    dTop:     { name: 'Dvojitý vrchol', short: 'DV', dir: -1, kind: 'g', txt: 'dva podobné vrcholy a prieraz pod dno medzi nimi' },
    hs:       { name: 'Hlava a ramená', short: 'HR', dir: -1, kind: 'g', txt: 'tri vrcholy, stredný najvyšší, a prieraz krčnej línie' },
    ihs:      { name: 'Obrátená hlava a ramená', short: 'OHR', dir: 1, kind: 'g', txt: 'tri dná, stredné najnižšie, a prieraz krčnej línie' },
    brkUp:    { name: 'Prieraz nad odpor', short: 'P↑', dir: 1, kind: 'g', txt: 'cena vyšla nad hornú hranicu pásma' },
    brkDn:    { name: 'Prieraz pod podporu', short: 'P↓', dir: -1, kind: 'g', txt: 'cena padla pod dolnú hranicu pásma' },
    flagUp:   { name: 'Býčia vlajka', short: 'VL↑', dir: 1, kind: 'g', txt: 'prudký rast, krátke vydýchnutie a prieraz hore' },
    flagDn:   { name: 'Medvedia vlajka', short: 'VL↓', dir: -1, kind: 'g', txt: 'prudký pokles, krátke vydýchnutie a prieraz dole' },
    triUp:    { name: 'Trojuholník, prieraz hore', short: 'T↑', dir: 1, kind: 'g', txt: 'zužujúce sa pásmo a cena vyšla hore' },
    triDn:    { name: 'Trojuholník, prieraz dole', short: 'T↓', dir: -1, kind: 'g', txt: 'zužujúce sa pásmo a cena vyšla dole' }
  };
  E.PATTERNS = PAT;

  function atrSeries(S, n) {
    var a = new Float64Array(n), i, s = 0, m = Math.min(14, n - 1), v;
    function tr(k) { return Math.max(S.h[k] - S.l[k], Math.abs(S.h[k] - S.c[k - 1]), Math.abs(S.l[k] - S.c[k - 1])); }
    for (i = 1; i <= m; i++) s += tr(i);
    v = m > 0 ? s / m : (S.h[0] - S.l[0]);
    for (i = 0; i < n; i++) {
      if (i > m) v += (tr(i) - v) / 14;
      a[i] = Math.max(v, S.c[i] * 1e-5);
    }
    return a;
  }

  // One pass over the candles: zigzag swing points (confirmed after a 2 ATR reversal), candle
  // patterns, chart patterns and three model inputs per candle.
  function detectPatterns(S, dt) {
    var n = S.c.length, A = atrSeries(S, n), ev = [], piv = [], pv = [], i, j, k;
    var seg = new Int32Array(n), fC = new Float64Array(n), fG = new Float64Array(n), fSR = new Float64Array(n);
    for (i = 1; i < n; i++) seg[i] = (dt && S.t[i] - S.t[i - 1] !== dt) ? i : seg[i - 1];
    var lastFire = {}, used = {};
    function fire(key, at, x) {
      var e = { i: at, t: S.t[at], key: key, dir: PAT[key].dir, kind: PAT[key].kind, px: S.c[at] };
      if (x) for (var q in x) e[q] = x[q];
      ev.push(e); lastFire[key] = at;
    }
    function cool(key, at, gap) { return lastFire[key] === undefined || at - lastFire[key] > gap; }
    var mode = 0, hiP = 0, hiI = 0, loP = 0, loI = 0;
    for (i = 0; i < n; i++) {
      var s0 = seg[i], a = A[i];
      if (i === s0) { mode = 0; hiP = S.h[i]; hiI = i; loP = S.l[i]; loI = i; pv = []; continue; }
      // ---- zigzag
      var th = 2 * a;
      if (mode >= 0 && S.h[i] >= hiP) { hiP = S.h[i]; hiI = i; }
      if (mode <= 0 && S.l[i] <= loP) { loP = S.l[i]; loI = i; }
      var np = null;
      if (mode === 0) {
        if (hiP - S.l[i] >= th && hiI < i) { np = { type: 1, i: hiI, p: hiP, conf: i }; mode = -1; loP = S.l[i]; loI = i; }
        else if (S.h[i] - loP >= th && loI < i) { np = { type: -1, i: loI, p: loP, conf: i }; mode = 1; hiP = S.h[i]; hiI = i; }
      } else if (mode === 1) {
        if (hiP - S.l[i] >= th) { np = { type: 1, i: hiI, p: hiP, conf: i }; mode = -1; loP = S.l[i]; loI = i; }
      } else if (S.h[i] - loP >= th) { np = { type: -1, i: loI, p: loP, conf: i }; mode = 1; hiP = S.h[i]; hiI = i; }
      if (np) { pv.push(np); piv.push(np); }
      var o = S.o[i], c = S.c[i], h = S.h[i], l = S.l[i], rng = h - l, body = Math.abs(c - o), top = Math.max(o, c), bot = Math.min(o, c);
      // ---- candle patterns (need 12 candles of the same run)
      if (i - s0 >= 12) {
        var up = h - top, lo = bot - l, tr1 = (S.c[i - 1] - S.c[i - 9]) / A[i - 1], tr3 = (S.c[i - 3] - S.c[i - 11]) / A[i - 3];
        if (rng >= 0.8 * a) {
          if (lo >= 0.6 * rng && up <= 0.15 * rng && body <= 0.35 * rng && tr1 <= -1 && cool('hammer', i, 3)) fire('hammer', i, { stop: l, lines: [] });
          if (up >= 0.6 * rng && lo <= 0.15 * rng && body <= 0.35 * rng && tr1 >= 1 && cool('star', i, 3)) fire('star', i, { stop: h, lines: [] });
          if (body <= 0.08 * rng && lastFire.hammer !== i && lastFire.star !== i && cool('doji', i, 3)) fire('doji', i, null);
        }
        var po = S.o[i - 1], pcl = S.c[i - 1], pb = Math.abs(pcl - po);
        if (pcl < po && c > o && c >= po && o <= pcl + 0.05 * a && body >= Math.max(1.05 * pb, 0.6 * a) && tr1 <= -0.5 && cool('engBull', i, 3)) fire('engBull', i, { stop: Math.min(l, S.l[i - 1]) });
        if (pcl > po && c < o && c <= po && o >= pcl - 0.05 * a && body >= Math.max(1.05 * pb, 0.6 * a) && tr1 >= 0.5 && cool('engBear', i, 3)) fire('engBear', i, { stop: Math.max(h, S.h[i - 1]) });
        var b2 = S.c[i - 2] - S.o[i - 2], b1 = Math.abs(S.c[i - 1] - S.o[i - 1]), mid2 = (S.o[i - 2] + S.c[i - 2]) / 2;
        if (b2 <= -0.8 * A[i - 2] && b1 <= 0.3 * -b2 && c > o && c >= mid2 && tr3 <= -1 && cool('mStar', i, 3)) fire('mStar', i, { stop: Math.min(S.l[i - 1], S.l[i - 2], l) });
        if (b2 >= 0.8 * A[i - 2] && b1 <= 0.3 * b2 && c < o && c <= mid2 && tr3 >= 1 && cool('eStar', i, 3)) fire('eStar', i, { stop: Math.max(S.h[i - 1], S.h[i - 2], h) });
        var sold = true, crow = true;
        for (j = i - 2; j <= i; j++) {
          var bj = S.c[j] - S.o[j], rj = S.h[j] - S.l[j] || 1e-12;
          if (!(bj >= 0.5 * A[j] && S.c[j] > S.c[j - 1] && S.h[j] - S.c[j] <= 0.35 * rj)) sold = false;
          if (!(bj <= -0.5 * A[j] && S.c[j] < S.c[j - 1] && S.c[j] - S.l[j] <= 0.35 * rj)) crow = false;
        }
        if (sold && tr3 <= 0 && cool('soldiers', i, 3)) fire('soldiers', i, { stop: S.o[i - 2] });
        if (crow && tr3 >= 0 && cool('crows', i, 3)) fire('crows', i, { stop: S.o[i - 2] });
      }
      // ---- chart patterns from the swing points of this run
      var L = pv.length, p1, p2, p3, p4, p5, nl, key, wid;
      if (L >= 3) {
        p1 = pv[L - 3]; p2 = pv[L - 2]; p3 = pv[L - 1]; wid = p3.i - p1.i;
        if (wid >= 6 && wid <= 150 && i > p3.i && i - p3.i <= Math.max(20, wid) && Math.abs(p1.p - p3.p) <= 0.75 * a) {
          nl = p2.p; key = (p1.type > 0 ? 'dTop' : 'dBot') + p3.i;
          if (!used[key]) {
            if (p1.type > 0 && c < nl && (S.c[i - 1] >= nl || i === p3.conf)) {
              used[key] = 1; var tp = Math.max(p1.p, p3.p);
              fire('dTop', i, { target: nl - (tp - nl), stop: tp, lines: [{ t1: S.t[p1.i], p1: nl, t2: S.t[i], p2: nl }, { t1: S.t[p1.i], p1: p1.p, t2: S.t[p3.i], p2: p3.p }] });
            } else if (p1.type < 0 && c > nl && (S.c[i - 1] <= nl || i === p3.conf)) {
              used[key] = 1; var bt = Math.min(p1.p, p3.p);
              fire('dBot', i, { target: nl + (nl - bt), stop: bt, lines: [{ t1: S.t[p1.i], p1: nl, t2: S.t[i], p2: nl }, { t1: S.t[p1.i], p1: p1.p, t2: S.t[p3.i], p2: p3.p }] });
            }
          }
        }
      }
      if (L >= 5) {
        p1 = pv[L - 5]; p2 = pv[L - 4]; p3 = pv[L - 3]; p4 = pv[L - 2]; p5 = pv[L - 1]; wid = p5.i - p1.i;
        var sgn = p1.type;  // +1: head and shoulders (tops), -1: inverse
        if (wid <= 240 && i > p5.i && i - p5.i <= Math.max(20, wid) &&
            sgn * (p3.p - p1.p) >= 0.5 * a && sgn * (p3.p - p5.p) >= 0.5 * a && Math.abs(p1.p - p5.p) <= 1.5 * a && p4.i > p2.i) {
          var nlAt = function (x) { return p2.p + (p4.p - p2.p) * (x - p2.i) / (p4.i - p2.i); };
          key = 'hs' + sgn + '_' + p5.i;
          var nNow = nlAt(i), nPrev = nlAt(i - 1);
          if (!used[key] && (sgn > 0 ? (c < nNow && (S.c[i - 1] >= nPrev || i === p5.conf)) : (c > nNow && (S.c[i - 1] <= nPrev || i === p5.conf)))) {
            used[key] = 1;
            var depth = Math.abs(p3.p - nlAt(p3.i));
            fire(sgn > 0 ? 'hs' : 'ihs', i, { target: nNow - sgn * depth, stop: p5.p, lines: [{ t1: S.t[p2.i], p1: p2.p, t2: S.t[i], p2: nNow }] });
          }
        }
      }
      // range breakout: tight 24-candle box, first close outside it
      if (i - s0 >= 25) {
        var rh = -Infinity, rl = Infinity;
        for (j = i - 24; j < i; j++) { if (S.h[j] > rh) rh = S.h[j]; if (S.l[j] < rl) rl = S.l[j]; }
        var bw = rh - rl, aw = A[i - 1];
        if (bw <= 4 * aw && bw >= 0.5 * aw) {
          if (c > rh + 0.25 * a && S.c[i - 1] <= rh && cool('brkUp', i, 12)) fire('brkUp', i, { target: rh + bw, stop: (rh + rl) / 2, lines: [{ t1: S.t[i - 24], p1: rh, t2: S.t[i], p2: rh }, { t1: S.t[i - 24], p1: rl, t2: S.t[i - 1], p2: rl }] });
          if (c < rl - 0.25 * a && S.c[i - 1] >= rl && cool('brkDn', i, 12)) fire('brkDn', i, { target: rl - bw, stop: (rh + rl) / 2, lines: [{ t1: S.t[i - 24], p1: rl, t2: S.t[i], p2: rl }, { t1: S.t[i - 24], p1: rh, t2: S.t[i - 1], p2: rh }] });
        }
      }
      // flags: strong pole, 4-15 candle pause that retraces at most half, close out of the pause
      if (i - s0 >= 30 && (cool('flagUp', i, 8) || cool('flagDn', i, 8))) {
        var fh = -Infinity, fl = Infinity;
        for (var f = 1; f <= 15; f++) {
          j = i - f; if (S.h[j] > fh) fh = S.h[j]; if (S.l[j] < fl) fl = S.l[j];
          if (f < 4) continue;
          var pe = j - 1; if (pe - 10 < s0) break;
          var pTop = Math.max(S.h[pe], S.h[pe - 1]), pBot = Math.min(S.l[pe], S.l[pe - 1]), lowest = Infinity, highest = -Infinity;
          for (k = pe - 10; k <= pe - 3; k++) { if (S.l[k] < lowest) lowest = S.l[k]; if (S.h[k] > highest) highest = S.h[k]; }
          var poleU = pTop - lowest, poleD = highest - pBot;
          if (poleU >= 4 * A[pe] && fh <= pTop + 0.2 * a && pTop - fl <= 0.5 * poleU && fh - fl <= 0.6 * poleU && c > fh && S.c[i - 1] <= fh && cool('flagUp', i, 8)) {
            fire('flagUp', i, { target: fh + poleU, stop: fl, lines: [{ t1: S.t[i - f], p1: fh, t2: S.t[i], p2: fh }] }); break;
          }
          if (poleD >= 4 * A[pe] && fl >= pBot - 0.2 * a && fh - pBot <= 0.5 * poleD && fh - fl <= 0.6 * poleD && c < fl && S.c[i - 1] >= fl && cool('flagDn', i, 8)) {
            fire('flagDn', i, { target: fl - poleD, stop: fh, lines: [{ t1: S.t[i - f], p1: fl, t2: S.t[i], p2: fl }] }); break;
          }
        }
      }
      // triangles from the last two swing highs and lows
      if (L >= 4) {
        var hs2 = [], ls2 = [];
        for (j = L - 4; j < L; j++) (pv[j].type > 0 ? hs2 : ls2).push(pv[j]);
        if (hs2.length === 2 && ls2.length === 2) {
          var Ha = hs2[0], Hb = hs2[1], La = ls2[0], Lb = ls2[1], first = Math.min(Ha.i, La.i), lastP = Math.max(Hb.i, Lb.i), span = lastP - first;
          var fallH = Hb.p <= Ha.p - 0.3 * a, flatH = Math.abs(Hb.p - Ha.p) < 0.3 * a, riseL = Lb.p >= La.p + 0.3 * a, flatL = Math.abs(Lb.p - La.p) < 0.3 * a;
          if (span >= 10 && span <= 160 && i > lastP && i - lastP <= span && ((fallH && riseL) || (flatH && riseL) || (fallH && flatL))) {
            var su = (Hb.p - Ha.p) / (Hb.i - Ha.i), sl = (Lb.p - La.p) / (Lb.i - La.i);
            var upAt = function (x) { return Ha.p + su * (x - Ha.i); }, loAt = function (x) { return La.p + sl * (x - La.i); };
            var uN = upAt(i), lN = loAt(i), hgt = upAt(first) - loAt(first), kind3 = fallH && riseL ? 'symetrický' : (flatH ? 'stúpajúci' : 'klesajúci');
            key = 'tri' + Hb.i + '_' + Lb.i;
            if (!used[key] && uN > lN && hgt > 0) {
              var tl = [{ t1: S.t[Ha.i], p1: Ha.p, t2: S.t[i], p2: uN }, { t1: S.t[La.i], p1: La.p, t2: S.t[i], p2: lN }];
              if (c > uN && S.c[i - 1] <= upAt(i - 1)) { used[key] = 1; fire('triUp', i, { target: uN + hgt, stop: lN, sub: kind3, lines: tl }); }
              else if (c < lN && S.c[i - 1] >= loAt(i - 1)) { used[key] = 1; fire('triDn', i, { target: lN - hgt, stop: uN, sub: kind3, lines: tl }); }
            }
          }
        }
      }
      // ---- model inputs: recent pattern signals and where the price sits between support and resistance
      var res = Infinity, sup = -Infinity;
      for (j = Math.max(0, L - 20); j < L; j++) {
        var pp = pv[j];
        if (pp.type > 0 && pp.p > c && pp.p < res) res = pp.p;
        if (pp.type < 0 && pp.p < c && pp.p > sup) sup = pp.p;
      }
      fSR[i] = isFinite(res) && isFinite(sup) ? clip(2 * (c - sup) / (res - sup) - 1, -1.5, 1.5) : (isFinite(res) ? -1.2 : (isFinite(sup) ? 1.2 : 0));
    }
    var wc = [1, 0.6, 0.35];
    ev.forEach(function (e) {
      if (!e.dir) return;
      if (e.kind === 'c') for (k = 0; k < 3 && e.i + k < n; k++) fC[e.i + k] += e.dir * wc[k];
      else for (k = 0; k < 6 && e.i + k < n; k++) fG[e.i + k] += e.dir * (1 - k / 6);
    });
    return { ev: ev, piv: piv, A: A, seg: seg, fC: fC, fG: fG, fSR: fSR };
  }

  // How often each pattern was followed by a move in its direction H candles later, against the
  // plain base rate of the same data. Shrunk towards the base rate so a few lucky cases do not count.
  function patternStats(ev, S, H, from, last) {
    var up = 0, tot = 0, i;
    for (i = from; i + H <= last; i++) { tot++; if (S.c[i + H] > S.c[i]) up++; }
    var p0 = tot ? up / tot : 0.5, st = {};
    Object.keys(PAT).forEach(function (k) { st[k] = { n: 0, k: 0, ret: 0, pend: 0, up: 0 }; });
    ev.forEach(function (e) {
      var s = st[e.key];
      if (e.i + H > last) { s.pend++; return; }
      var r = Math.log(S.c[e.i + H] / S.c[e.i]);
      e.out = r; s.n++; if (r > 0) s.up++;
      s.ret += (e.dir || 1) * r;
      if (e.dir > 0 ? r > 0 : (e.dir < 0 ? r < 0 : false)) s.k++;
    });
    Object.keys(st).forEach(function (k) {
      var s = st[k], d = PAT[k].dir, base = d >= 0 ? p0 : 1 - p0;
      if (!d) s.k = s.up;
      s.base = base; s.rate = s.n ? s.k / s.n : NaN;
      s.shr = (s.k + 20 * base) / (s.n + 20);
      s.edge = s.shr - base;
      s.z = s.n ? (s.k - s.n * base) / Math.sqrt(s.n * base * (1 - base)) : 0;
      s.avg = s.n ? s.ret / s.n : 0;
      s.verdict = !d ? 'info' : (s.n < 10 ? 'few' : (s.z >= 1.64 && s.edge >= 0.03 ? 'works' : (s.z <= -1.64 && s.edge <= -0.03 ? 'reverse' : 'weak')));
    });
    return { p0: p0, nBase: tot, st: st };
  }

  // Current reading: patterns that just fired (candle patterns in the last 3 candles, chart
  // patterns in the last 6) and are not broken yet, combined into one probability of a rise.
  function patternAdvice(P, stats, S, last) {
    var act = [], i, e;
    for (var q = P.ev.length - 1; q >= 0; q--) {
      e = P.ev[q];
      if (last - e.i > 6) break;
      if (!e.dir || (e.kind === 'c' && last - e.i > 2)) continue;
      var broken = false;
      if (isFinite(e.stop)) for (i = e.i + 1; i <= last; i++) if (e.dir > 0 ? S.c[i] < e.stop : S.c[i] > e.stop) { broken = true; break; }
      if (!broken) act.push(e);
    }
    var p0 = stats.p0, z = logit(p0), m = act.length;
    act.forEach(function (x) {
      var s = stats.st[x.key], pu = x.dir > 0 ? s.shr : 1 - s.shr;
      z += (logit(pu) - logit(p0)) / Math.sqrt(m);
    });
    act.sort(function (a, b) { return (b.kind === 'g') - (a.kind === 'g') || Math.abs(stats.st[b.key].edge) - Math.abs(stats.st[a.key].edge) || b.i - a.i; });
    var prev = null;
    for (q = P.ev.length - 1; q >= 0; q--) if (P.ev[q].dir && act.indexOf(P.ev[q]) < 0) { prev = P.ev[q]; break; }
    // nearest swing support and resistance of the newest run
    var c = S.c[last], res = Infinity, sup = -Infinity, nRun = 0;
    for (q = P.piv.length - 1; q >= 0 && nRun < 30; q--) {
      var pp = P.piv[q];
      if (pp.i < P.seg[last]) break;
      nRun++;
      if (pp.type > 0 && pp.p > c && pp.p < res) res = pp.p;
      if (pp.type < 0 && pp.p < c && pp.p > sup) sup = pp.p;
    }
    return { active: act, lead: act[0] || null, p: m ? sigmoid(z) : p0, p0: p0, prev: prev,
      res: isFinite(res) ? res : null, sup: isFinite(sup) ? sup : null, ago: prev ? last - prev.i : null };
  }

  // ---------------------------------------------------------------- features
  var FEAT_NAMES = [
    'SOL posledná sviečka', 'SOL momentum 4', 'SOL momentum 16', 'SOL momentum 64',
    'BTC posledná sviečka', 'BTC momentum 16', 'SOL voči BTC (16)', 'Taker buy pomer (3)',
    'Objem z-skóre', 'Volatilita vs. dlhodobá',
    'Sviečkové patterny', 'Grafové patterny', 'Poloha medzi podporou a odporom'
  ];
  var D = FEAT_NAMES.length;
  E.FEAT_NAMES = FEAT_NAMES;
  var WARM = 160;

  function buildFeatures(S, B, rS, rB, P) {
    var n = S.c.length, X = new Float64Array(n * D), i, k;
    var lf = 0.97, ls = 0.997, nInit = Math.min(n - 1, 100), init = 0, initB = 0;
    for (i = 1; i <= nInit; i++) { init += rS[i] * rS[i]; initB += rB[i] * rB[i]; }
    init = init / nInit || 1e-6; initB = initB / nInit || 1e-6;
    var ef = init, es = init, eb = initB, csS = 0, csB = 0, cv = 0, vb = initB, beta = 1;
    var sf = new Float64Array(n), cS = new Float64Array(n), cB = new Float64Array(n);
    var lvM = Math.log(Math.max(S.v[0], 1e-9)), lvV = 1;
    for (i = 0; i < n; i++) {
      if (i > 0) {
        ef = lf * ef + (1 - lf) * rS[i] * rS[i];
        es = ls * es + (1 - ls) * rS[i] * rS[i];
        eb = lf * eb + (1 - lf) * rB[i] * rB[i];
        csS += rS[i]; csB += rB[i];
        cv = 0.99 * cv + 0.01 * rS[i] * rB[i]; vb = 0.99 * vb + 0.01 * rB[i] * rB[i];
        beta = cv / vb;
      }
      var s1 = Math.sqrt(ef) || 1e-6, b1 = Math.sqrt(eb) || 1e-6;
      sf[i] = s1; cS[i] = csS; cB[i] = csB;
      var lv = Math.log(Math.max(S.v[i], 1e-9)), dv = lv - lvM, volZ = dv / Math.sqrt(lvV);
      lvM += 0.02 * dv; lvV = 0.98 * lvV + 0.02 * dv * dv;
      if (i < 64) continue;
      var o = i * D;
      X[o] = rS[i] / s1;
      X[o + 1] = (cS[i] - cS[i - 4]) / (s1 * 2);
      X[o + 2] = (cS[i] - cS[i - 16]) / (s1 * 4);
      X[o + 3] = (cS[i] - cS[i - 64]) / (s1 * 8);
      X[o + 4] = rB[i] / b1;
      X[o + 5] = (cB[i] - cB[i - 16]) / (b1 * 4);
      X[o + 6] = ((cS[i] - cS[i - 16]) - beta * (cB[i] - cB[i - 16])) / (s1 * 4);
      var st = 0, sv = 0;
      for (k = i - 2; k <= i; k++) if (isFinite(S.tb[k]) && S.v[k] > 0) { st += S.tb[k]; sv += S.v[k]; }
      X[o + 7] = sv > 0 ? st / sv - 0.5 : 0;
      X[o + 8] = clip(volZ, -5, 5);
      X[o + 9] = Math.log(s1 / (Math.sqrt(es) || s1));
      X[o + 10] = P.fC[i]; X[o + 11] = P.fG[i]; X[o + 12] = P.fSR[i];
    }
    return { X: X, sf: sf };
  }

  var ALL = Array.from({ length: D }, function (_, j) { return j; });
  function featStats(X, a, b, idx) {
    idx = idx || ALL;
    var m = idx.length, mu = new Float64Array(m), sd = new Float64Array(m), n = b - a + 1, i, j;
    for (i = a; i <= b; i++) for (j = 0; j < m; j++) mu[j] += X[i * D + idx[j]];
    for (j = 0; j < m; j++) mu[j] /= n;
    for (i = a; i <= b; i++) for (j = 0; j < m; j++) { var d0 = X[i * D + idx[j]] - mu[j]; sd[j] += d0 * d0; }
    for (j = 0; j < m; j++) sd[j] = Math.sqrt(sd[j] / Math.max(1, n - 1)) || 1;
    return { mu: mu, sd: sd, idx: idx };
  }

  function rowVec(X, i, st, out) {
    var idx = st.idx;
    out[0] = 1;
    for (var j = 0; j < idx.length; j++) out[j + 1] = clip((X[i * D + idx[j]] - st.mu[j]) / st.sd[j], -4, 4);
    return out;
  }

  // L2 logistic regression by Newton/IRLS. Intercept not penalised.
  // Optional recency weighting: a row hl candles older than the newest counts half.
  function rowWeights(a, b, hl) {
    var rw = new Float64Array(b - a + 1), sum = 0;
    for (var i = a; i <= b; i++) { rw[i - a] = hl ? Math.pow(0.5, (b - i) / hl) : 1; sum += rw[i - a]; }
    return { w: rw, sum: sum };
  }
  function fitLogit(X, y, a, b, st, lambda, w0, rwt) {
    var P = st.idx.length + 1, w = w0 && w0.length === P ? Float64Array.from(w0) : new Float64Array(P), x = new Float64Array(P);
    var RW = rwt ? rwt.w : null;
    for (var it = 0; it < 25; it++) {
      var g = new Float64Array(P), Hm = new Float64Array(P * P), i, j, k;
      for (i = a; i <= b; i++) {
        rowVec(X, i, st, x);
        var z = 0;
        for (j = 0; j < P; j++) z += w[j] * x[j];
        var q0 = RW ? RW[i - a] : 1, p = sigmoid(z), err = (p - y[i]) * q0, s = p * (1 - p) * q0;
        for (j = 0; j < P; j++) {
          g[j] += err * x[j];
          var sx = s * x[j];
          for (k = 0; k <= j; k++) Hm[j * P + k] += sx * x[k];
        }
      }
      for (j = 0; j < P; j++) for (k = 0; k < j; k++) Hm[k * P + j] = Hm[j * P + k];
      for (j = 1; j < P; j++) { g[j] += lambda * w[j]; Hm[j * P + j] += lambda; }
      Hm[0] += 1e-6;
      var step = cholSolve(Hm, g, P), mx = 0;
      for (j = 0; j < P; j++) { w[j] -= step[j]; mx = Math.max(mx, Math.abs(step[j])); }
      if (mx < 1e-7) break;
    }
    return w;
  }
  function predLogit(X, i, st, w) {
    var P = st.idx.length + 1, x = rowVec(X, i, st, new Float64Array(P)), z = 0;
    for (var j = 0; j < P; j++) z += w[j] * x[j];
    return sigmoid(z);
  }

  // Experts the app learns to trust: each one is a logistic model on a subset of the inputs;
  // the last one always says 50 % (it wins when nobody predicts anything).
  var BASEIDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  function patIdx(cfg) { return !cfg.patKeys || cfg.sr ? [10, 11, 12] : [10, 11]; }
  function expertsFor(cfg) {
    var out = [
      { key: 'all', name: 'Kombinovaný (všetko)', idx: cfg.pat ? BASEIDX.concat(patIdx(cfg)) : BASEIDX },
      { key: 'sol', name: 'Momentum SOL', idx: [0, 1, 2, 3] },
      { key: 'btc', name: 'Bitcoin predstih', idx: [4, 5, 6] },
      { key: 'flow', name: 'Objem a order flow', idx: [7, 8, 9] }
    ];
    if (cfg.pat) out.push({ key: 'pat', name: 'Naučené patterny', idx: patIdx(cfg) });
    out.push({ key: 'base', name: 'Náhoda 50 %', idx: null });
    return out;
  }
  // Model variants the app compares every few days (window and recency in candles). The main model
  // does not use patterns; they enter it only after "pattern learning" adopted a set of them.
  var DEFAULT_CFG = { id: 'long', win: 3000, hl: 0, pat: false, lamX: 1 };
  E.DEFAULT_CFG = DEFAULT_CFG;
  E.CANDIDATES = [
    DEFAULT_CFG,
    { id: 'longRecent', win: 3000, hl: 500, pat: false, lamX: 1 },
    { id: 'mid', win: 2000, hl: 0, pat: false, lamX: 1 },
    { id: 'short', win: 1000, hl: 0, pat: false, lamX: 1 },
    { id: 'shortRecent', win: 1500, hl: 250, pat: false, lamX: 2 },
    { id: 'careful', win: 3000, hl: 0, pat: false, lamX: 3 }
  ];
  function normCfg(c) {
    c = c || {};
    var keys = Array.isArray(c.patKeys) ? c.patKeys.filter(function (p) { return p && PAT[p.key] && PAT[p.key].dir; }).map(function (p) { return { key: p.key, sign: p.sign < 0 ? -1 : 1 }; }) : null;
    return { id: c.id || 'custom', win: c.win || 3000, hl: c.hl || 0, pat: !!c.pat, patKeys: keys, sr: !!c.sr, lamX: c.lamX || 1 };
  }
  E.normCfg = normCfg;
  // Inputs for a variant: the chosen patterns only (with their sign), support/resistance on request.
  function featX(ctx, cfg) {
    if (!cfg.pat || !cfg.patKeys) return ctx.X;
    var sig = cfg.patKeys.map(function (p) { return p.key + (p.sign < 0 ? '-' : '+'); }).sort().join(',') + (cfg.sr ? '|sr' : '');
    ctx.xc = ctx.xc || {};
    if (ctx.xc[sig]) return ctx.xc[sig];
    var n = ctx.n, X2 = Float64Array.from(ctx.X), sel = {}, i, k, wc = [1, 0.6, 0.35];
    cfg.patKeys.forEach(function (p) { sel[p.key] = p.sign; });
    for (i = 0; i < n; i++) { X2[i * D + 10] = 0; X2[i * D + 11] = 0; if (!cfg.sr) X2[i * D + 12] = 0; }
    ctx.P.ev.forEach(function (ev) {
      var s = sel[ev.key];
      if (!s || !ev.dir) return;
      if (ev.kind === 'c') for (k = 0; k < 3 && ev.i + k < n; k++) X2[(ev.i + k) * D + 10] += s * ev.dir * wc[k];
      else for (k = 0; k < 6 && ev.i + k < n; k++) X2[(ev.i + k) * D + 11] += s * ev.dir * (1 - k / 6);
    });
    return (ctx.xc[sig] = X2);
  }
  var EXPERTS = expertsFor(DEFAULT_CFG);
  // candles needed before the direction model is trained and tested (below that: range only)
  E.minFull = function (H) { return 731 + 2 * H; };
  E.EXPERTS = EXPERTS.map(function (x) { return { key: x.key, name: x.name }; });
  E.expertsFor = function (cfg) { return expertsFor(normCfg(cfg)).map(function (x) { return { key: x.key, name: x.name }; }); };
  function logloss(p, y) { p = Math.min(1 - 1e-4, Math.max(1e-4, p)); return y ? -Math.log(p) : -Math.log(1 - p); }
  // Hedge weights from discounted losses (one effective observation per H overlapping labels).
  function hedge(L, eta) {
    var mn = Math.min.apply(null, L), w = L.map(function (l) { return Math.exp(-eta * (l - mn)); }), s = w.reduce(function (a, b) { return a + b; }, 0);
    return w.map(function (x) { return x / s; });
  }

  // Platt scaling on standardised logit(p_raw); observation weight 1/H for overlapping labels;
  // slope constrained >= 0 (a model that is wrong out-of-sample is treated as no skill);
  // strong prior on the intercept so a few trending days do not turn into a LONG/SHORT call.
  function fitPlatt(pRaw, y, n, H) {
    var m = 0, s = 0, i;
    var lg = new Float64Array(n);
    for (i = 0; i < n; i++) { lg[i] = logit(pRaw[i]); m += lg[i]; }
    m /= n;
    for (i = 0; i < n; i++) s += (lg[i] - m) * (lg[i] - m);
    var sd = Math.sqrt(s / Math.max(1, n - 1)) || 1, wt = 1 / Math.max(1, H), lamA = 30, lamB = 1;
    function solve(useSlope) {
      var a = 0, b = 0;
      for (var it = 0; it < 30; it++) {
        var g0 = lamA * a, g1 = lamB * b, h00 = lamA, h01 = 0, h11 = lamB;
        for (var q = 0; q < n; q++) {
          var x = (lg[q] - m) / sd, p = sigmoid(a + (useSlope ? b * x : 0)), e = (p - y[q]) * wt, ss = p * (1 - p) * wt;
          g0 += e; h00 += ss;
          if (useSlope) { g1 += e * x; h01 += ss * x; h11 += ss * x * x; }
        }
        var da, db = 0;
        if (useSlope) { var det = h00 * h11 - h01 * h01; da = (h11 * g0 - h01 * g1) / det; db = (h00 * g1 - h01 * g0) / det; }
        else da = g0 / h00;
        a -= da; b -= db;
        if (Math.abs(da) + Math.abs(db) < 1e-8) break;
      }
      return { a: a, b: b };
    }
    var r = solve(true);
    if (r.b < 0) r = solve(false);
    return { a: r.a, b: r.b, m: m, sd: sd };
  }
  function applyPlatt(c, p) { return sigmoid(c.a + c.b * (logit(p) - c.m) / c.sd); }

  // ---------------------------------------------------------------- simulation
  function makePool(PS, PB, from, to) {
    var idx = [], i;
    for (i = Math.max(2, from); i < to; i++) if (isFinite(PS.z[i]) && isFinite(PB.z[i])) idx.push(i);
    var n = idx.length, mS = 0, mB = 0, vS = 0, vB = 0;
    for (i = 0; i < n; i++) { mS += PS.z[idx[i]]; mB += PB.z[idx[i]]; }
    mS /= n; mB /= n;
    for (i = 0; i < n; i++) { vS += Math.pow(PS.z[idx[i]] - mS, 2); vB += Math.pow(PB.z[idx[i]] - mB, 2); }
    return { idx: Int32Array.from(idx), mS: mS, mB: mB, sS: Math.sqrt(vS / n) || 1, sB: Math.sqrt(vB / n) || 1 };
  }

  // Joint SOL/BTC shock paths (no drift). futB: buckets of the future bars.
  function simShocks(o) {
    var M = o.M, H = o.H, PS = o.PS, PB = o.PB, pool = o.pool, rnd = o.rnd, n = pool.idx.length;
    var shS = new Float64Array(M * H), sgS = new Float64Array(M * H), uS = new Float64Array(M * H), wS = new Float64Array(M * H);
    var shB = o.withBtc ? new Float64Array(M * H) : null;
    var ps = PS.gp, pb = PB.gp, sc = o.scale || 1;
    for (var m = 0; m < M; m++) {
      var hs = o.hs0, hb = o.hb0;
      for (var k = 0; k < H; k++) {
        var j = pool.idx[(rnd() * n) | 0], q = m * H + k;
        var eS = Math.sqrt(hs) * (PS.z[j] - pool.mS) / pool.sS;
        var sea = PS.season[o.futB[k]];
        shS[q] = sea * eS * sc; sgS[q] = sea * Math.sqrt(hs) * sc; uS[q] = PS.U[j]; wS[q] = PS.W[j];
        hs = ps.omega + ps.alpha * eS * eS + ps.beta * hs;
        if (shB) {
          var eB = Math.sqrt(hb) * (PB.z[j] - pool.mB) / pool.sB;
          shB[q] = PB.season[o.futB[k]] * eB * sc;
          hb = pb.omega + pb.alpha * eB * eB + pb.beta * hb;
        }
      }
    }
    return { shS: shS, sgS: sgS, uS: uS, wS: wS, shB: shB, M: M, H: H };
  }

  // Drift so that P(sum of log returns > 0) = pUp, spread across steps by expected variance.
  function driftFor(sim, pUp) {
    var M = sim.M, H = sim.H, R = new Float64Array(M), wk = new Float64Array(H), tot = 0, m, k;
    for (m = 0; m < M; m++) { var s = 0; for (k = 0; k < H; k++) { s += sim.shS[m * H + k]; wk[k] += sim.sgS[m * H + k] * sim.sgS[m * H + k]; } R[m] = s; }
    var sorted = Float64Array.from(R).sort();
    var Dt = -qSorted(sorted, 1 - pUp);
    for (k = 0; k < H; k++) tot += wk[k];
    var dk = new Float64Array(H);
    for (k = 0; k < H; k++) dk[k] = Dt * wk[k] / tot;
    return { D: Dt, dk: dk, R: R, sorted: sorted };
  }

  function futureBuckets(tLast, dt, H) {
    var b = new Uint8Array(H);
    for (var k = 0; k < H; k++) b[k] = bucketOf(tLast + (k + 1) * dt);
    return b;
  }

  // ---------------------------------------------------------------- main run
  // Everything that does not depend on the model variant: aligned data, volatility, patterns, inputs.
  function prepare(sol, btc, opt) {
    var interval = opt.interval || '15m', dt = INTERVAL_MS[interval], H = opt.H || 24, i;
    var Al = E.align(sol, btc), S = Al.s, B = Al.b, n = Al.n;
    if (n < 40) throw new Error('Málo dát: ' + n + ' spoločných sviečok (treba aspoň 40).');
    var PS = prepSym(S, dt, n), PB = prepSym(B, dt, n), P = detectPatterns(S, dt);
    var X = buildFeatures(S, B, PS.r, PB.r, P).X;
    var y = new Uint8Array(n);
    for (i = 0; i + H < n; i++) y[i] = S.c[i + H] > S.c[i] ? 1 : 0;
    return { interval: interval, dt: dt, H: H, S: S, B: B, n: n, PS: PS, PB: PB, P: P, X: X, y: y, warm: n >= 1500 ? WARM : 80 };
  }
  E.prepare = function (sol, btc, opt) { return prepare(sol, btc, opt || {}); };

  // Walk-forward of the direction experts of one model variant over the points [evalStart, evalEnd]:
  // refit every R candles on purged history only, expanding Platt calibration, online Hedge weights.
  function direction(ctx, cfg, evalStart, evalEnd) {
    var H = ctx.H, X = featX(ctx, cfg), y = ctx.y, S = ctx.S, warm = ctx.warm, R = Math.max(12, H), Wtr = cfg.win;
    var EXP = expertsFor(cfg), NX = EXP.length, Ne = evalEnd - evalStart + 1, e, i, k, x;
    var lam = function (rows) { return rows * 0.05 * (1 + H / 8) * cfg.lamX; };
    var pRawX = [], pCalX = [], wX = [], stX = [], pRef = new Float64Array(Ne), ref = 0.5;
    for (x = 0; x < NX; x++) { pRawX.push(new Float64Array(Ne)); pCalX.push(new Float64Array(Ne)); wX.push(null); stX.push(null); }
    for (e = 0; e < Ne; e++) {
      i = evalStart + e;
      if (e % R === 0) {
        var b0 = i - H, a0 = Math.max(warm, b0 - Wtr + 1), rwt = rowWeights(a0, b0, cfg.hl);
        for (x = 0; x < NX; x++) {
          if (!EXP[x].idx) continue;
          stX[x] = featStats(X, a0, b0, EXP[x].idx);
          wX[x] = fitLogit(X, y, a0, b0, stX[x], lam(rwt.sum), wX[x], rwt);
        }
        var sy = 0; for (k = a0; k <= b0; k++) sy += y[k];
        ref = sy / (b0 - a0 + 1);
      }
      for (x = 0; x < NX; x++) pRawX[x][e] = EXP[x].idx ? predLogit(X, i, stX[x], wX[x]) : 0.5;
      pRef[e] = ref;
    }
    var calMin = 150, calX = [], yE = new Uint8Array(Ne), retE = new Float64Array(Ne);
    for (x = 0; x < NX; x++) calX.push(null);
    for (e = 0; e < Ne; e++) { i = evalStart + e; yE[e] = y[i]; retE[e] = S.c[i + H] / S.c[i] - 1; }
    for (e = 0; e < Ne; e++) {
      if (e % R === 0) { var nres = e - H + 1; if (nres >= calMin) for (x = 0; x < NX; x++) if (EXP[x].idx) calX[x] = fitPlatt(pRawX[x], yE, nres, H); }
      for (x = 0; x < NX; x++) pCalX[x][e] = !EXP[x].idx ? 0.5 : (calX[x] ? applyPlatt(calX[x], pRawX[x][e]) : 0.5 + 0.5 * (pRawX[x][e] - 0.5));
    }
    // Online Hedge over the experts: the weight used at point e only knows outcomes resolved by then
    // (k <= e - H); older losses fade with a half-life of max(150, 6H) points.
    var gam = Math.pow(0.5, 1 / Math.max(150, 6 * H)), Lx = new Float64Array(NX), pCal = new Float64Array(Ne), next = 0;
    function resolveUpTo(lim) {
      while (next <= lim) {
        for (var q = 0; q < NX; q++) Lx[q] = gam * Lx[q] + logloss(pCalX[q][next], yE[next]) / H;
        next++;
      }
    }
    for (e = 0; e < Ne; e++) {
      resolveUpTo(e - H);
      var wts = hedge(Array.from(Lx), 1), pe = 0;
      for (x = 0; x < NX; x++) pe += wts[x] * pCalX[x][e];
      pCal[e] = pe;
    }
    resolveUpTo(Ne - 1);
    return { EXP: EXP, NX: NX, Ne: Ne, pRawX: pRawX, pCalX: pCalX, pCal: pCal, pRef: pRef, yE: yE, retE: retE, Lx: Lx, wX: wX, lam: lam, calMin: calMin };
  }

  // Scores one model variant on the newest K resolved points. Nothing it scores was used to fit it.
  E._detect = function (S, dt) { return detectPatterns(S, dt); };
  // Everything "pattern learning" weighs: per pattern all history, the newest window, older data,
  // and trending vs sideways markets (efficiency ratio over 48 candles at the event).
  E.patternEvidence = function (ctx, fromT) {
    var S = ctx.S, H = ctx.H, last = ctx.n - 1, P = ctx.P, st = patternStats(P.ev, S, H, 20, last), rows = {}, i;
    function er(j) {
      var Lb = Math.min(48, j);
      if (Lb < 10) return NaN;
      var s3 = 0;
      for (var q = j - Lb + 1; q <= j; q++) s3 += Math.abs(S.c[q] - S.c[q - 1]);
      return s3 > 0 ? Math.abs(S.c[j] - S.c[j - Lb]) / s3 : 0;
    }
    Object.keys(PAT).forEach(function (k) {
      var s = st.st[k];
      rows[k] = { key: k, name: PAT[k].name, dir: PAT[k].dir, kind: PAT[k].kind, n: s.n, k: s.k, rate: s.rate, base: s.base, edge: s.edge, z: s.z, avg: s.avg, verdict: s.verdict, pend: s.pend,
        n5: 0, k5: 0, nOld: 0, kOld: 0, nTr: 0, kTr: 0, nRg: 0, kRg: 0 };
    });
    P.ev.forEach(function (ev) {
      if (!ev.dir || ev.out == null || ev.i + H > last) return;
      var r = rows[ev.key], ok = ev.dir > 0 ? ev.out > 0 : ev.out < 0, x = er(ev.i);
      if (ev.t >= fromT) { r.n5++; if (ok) r.k5++; } else { r.nOld++; if (ok) r.kOld++; }
      if (x >= 0.35) { r.nTr++; if (ok) r.kTr++; } else if (x <= 0.2) { r.nRg++; if (ok) r.kRg++; }
    });
    var up5 = 0, n5 = 0;
    for (i = 20; i + H <= last; i++) if (S.t[i] >= fromT) { n5++; if (S.c[i + H] > S.c[i]) up5++; }
    return { H: H, p0: st.p0, p0recent: n5 ? up5 / n5 : NaN, nRecent: n5, erNow: er(last), fromT: fromT, toT: S.t[last],
      rows: Object.keys(rows).map(function (k) { return rows[k]; }).filter(function (r) { return r.dir; }) };
  };
  E._stats = function (ev, S, H, from, last) { return patternStats(ev, S, H, from, last); };
  E.evalConfig = function (ctx, cfg, K, theta) {
    cfg = normCfg(cfg);
    var H = ctx.H, evalEnd = ctx.n - 1 - H, minStart = ctx.warm + 400 + H;
    K = Math.min(K, evalEnd - minStart + 1 - 200);
    if (!(K >= 100)) return null;
    var evalStart = Math.max(minStart, evalEnd - (K + 300) + 1);
    var Dn = direction(ctx, cfg, evalStart, evalEnd), from = Dn.Ne - K, lls = new Float64Array(K);
    var ll = 0, br = 0, brR = 0, hits = 0, nz = 0, sigN = 0, sigHit = 0, th = theta == null ? 0.05 : theta;
    for (var e = from; e < Dn.Ne; e++) {
      var p = Dn.pCal[e], yy = Dn.yE[e], l = logloss(p, yy);
      lls[e - from] = l; ll += l; br += (p - yy) * (p - yy); brR += (Dn.pRef[e] - yy) * (Dn.pRef[e] - yy);
      if (p !== 0.5) { nz++; if ((p > 0.5) === (yy === 1)) hits++; }
      if (Math.abs(p - 0.5) >= th) { sigN++; if ((p > 0.5) === (yy === 1)) sigHit++; }
    }
    return { cfg: cfg, K: K, ll: ll / K, brier: br / K, bss: 1 - br / brR, hit: nz ? hits / nz : NaN, sigN: sigN, sigHit: sigN ? sigHit / sigN : NaN,
      lls: lls, t1: ctx.S.t[evalStart + from], t2: ctx.S.t[evalEnd] };
  };

  E.run = function (sol, btc, opt) {
    opt = opt || {};
    var t0 = Date.now(), log = [];
    function L(msg, ok) { log.push({ ms: Date.now() - t0, msg: msg, ok: ok !== false }); }
    var cfg = normCfg(opt.cfg), ctx = opt.ctx || prepare(sol, btc, opt);
    var interval = ctx.interval, dt = ctx.dt, H = ctx.H, Mlive = opt.M || 400;
    var seed = opt.seed != null ? opt.seed : 12345;
    var S = ctx.S, B = ctx.B, n = ctx.n, P = ctx.P;
    L('zarovnané SOL/BTC · ' + n + ' sviečok · ' + interval);
    var last = n - 1, i, e, k;

    // live volatility model on all data
    var PS = ctx.PS, PB = ctx.PB;
    L('GARCH(1,1) SOL α=' + PS.gp.alpha.toFixed(3) + ' β=' + PS.gp.beta.toFixed(3) + ' · BTC α=' + PB.gp.alpha.toFixed(3) + ' β=' + PB.gp.beta.toFixed(3));
    var X = featX(ctx, cfg), y = ctx.y;
    L('vstupy ' + expertsFor(cfg)[0].idx.length + ' (SOL + BTC + order flow' + (cfg.pat ? ' + naučené patterny' + (cfg.patKeys ? ' ' + cfg.patKeys.length : '') : '') + ') · patterny v histórii: ' + P.ev.length);

    // ---- walk-forward
    var warm = ctx.warm, Wtr = cfg.win, Wmin = 400, R = Math.max(12, H);
    var evalEnd = last - H;
    var evalStart = Math.max(warm + Wmin + H, evalEnd - (opt.wfMax || 1500) + 1);
    var Ne = evalEnd - evalStart + 1;
    var lite = Ne < 250;
    var wf = null, kappaLive = 1, pNow = 0.5, pRawNow = 0.5, calL = { b: 0 };
    var EXP = expertsFor(cfg), NX = EXP.length, x, lam = null;
    var contrib = EXP[0].idx.map(function (j) { return { name: FEAT_NAMES[j], w: 0, x: 0, c: 0 }; });
    var experts = EXP.map(function (o) { return { key: o.key, name: o.name, weight: o.idx ? 0 : 1, pNow: 0.5, bss: 0 }; });
    var bL = last - H, aL = Math.max(warm, bL - Wtr + 1);
    if (!lite) {
    var Dn = direction(ctx, cfg, evalStart, evalEnd), pRawX = Dn.pRawX, pCalX = Dn.pCalX, pCal = Dn.pCal, pRef = Dn.pRef;
    var yE = Dn.yE, retE = Dn.retE, Lx = Dn.Lx, wX = Dn.wX, calMin = Dn.calMin;
    lam = Dn.lam;
    var liveL = opt.liveLoss && opt.liveLoss.length === NX ? opt.liveLoss : null;
    var wFinal = hedge(Array.from(Lx).map(function (l, q) { return l + (liveL ? liveL[q] : 0); }), 1);
    var mFrom = Math.min(Ne - 50, calMin + H);
    L('walk-forward ' + Ne + ' bodov · ' + (NX - 1) + ' modely + náhoda · retrain každých ' + R + ' · purge ' + H);

    // ---- out-of-sample metrics (direction)
    var nM = 0, hits = 0, br = 0, brRef = 0, nz = 0;
    for (e = mFrom; e < Ne; e++) {
      var p = pCal[e], yy = yE[e];
      br += (p - yy) * (p - yy); brRef += (pRef[e] - yy) * (pRef[e] - yy); nM++;
      if (p !== 0.5) { nz++; if ((p > 0.5) === (yy === 1)) hits++; }
    }
    var nEff = Math.max(1, nM / H);
    var hit = hits / Math.max(1, nz);
    wf = {
      evalStart: evalStart, mFrom: mFrom, Ne: Ne, H: H, n: nM, nEff: nEff,
      hit: hit, hitCI: 1.96 * Math.sqrt(hit * (1 - hit) / nEff),
      brier: br / nM, brierRef: brRef / nM, bss: 1 - (br / nM) / (brRef / nM),
      pCal: pCal, pRaw: pRawX[0], y: yE, ret: retE, t: Float64Array.from({ length: Ne }, function (_, q) { return S.t[evalStart + q]; })
    };
    for (x = 0; x < NX; x++) {
      var bx = 0;
      for (e = mFrom; e < Ne; e++) bx += Math.pow(pCalX[x][e] - yE[e], 2);
      experts[x].bss = 1 - (bx / nM) / (brRef / nM);
    }

    // ---- out-of-sample coverage of the FHS bands (vol model fitted before evalStart)
    var PSe = prepSym(S, dt, evalStart), PBe = prepSym(B, dt, evalStart);
    var rndE = mulberry32(seed ^ 0x9e3779b9), Me = 150, cs = Math.max(1, Math.round(H / 8));
    var c80 = 0, c50 = 0, c1 = 0, nc = 0, raw80 = 0, pit = new Float64Array(10), sc80L = [], sc50L = [];
    function kappa(list, maxE, q) {
      var v = [];
      for (var a2 = 0; a2 < list.length; a2++) if (list[a2].e <= maxE) v.push(list[a2].s);
      return v.length < 30 ? 1 : clip(quantile(v, q), 0.6, 1.8);
    }
    for (e = mFrom; e < Ne; e += cs) {
      i = evalStart + e;
      var pool = makePool(PSe, PBe, i - 1500, i + 1);
      var sim = simShocks({ M: Me, H: H, PS: PSe, PB: PBe, pool: pool, rnd: rndE, hs0: PSe.h[i + 1], hb0: PBe.h[i + 1], futB: futureBuckets(S.t[i], dt, H), withBtc: false });
      var dr = driftFor(sim, pCal[e]);
      var act = Math.log(S.c[i + H] / S.c[i]);
      var q10 = qSorted(dr.sorted, 0.1) + dr.D, q90 = qSorted(dr.sorted, 0.9) + dr.D;
      var q25 = qSorted(dr.sorted, 0.25) + dr.D, q75 = qSorted(dr.sorted, 0.75) + dr.D, q50 = qSorted(dr.sorted, 0.5) + dr.D;
      // online conformal width calibration: only scores already resolved at time i are used
      var s80 = act >= q50 ? (act - q50) / Math.max(1e-12, q90 - q50) : (q50 - act) / Math.max(1e-12, q50 - q10);
      var s50 = act >= q50 ? (act - q50) / Math.max(1e-12, q75 - q50) : (q50 - act) / Math.max(1e-12, q50 - q25);
      var k80 = kappa(sc80L, e - H, 0.8), k50 = kappa(sc50L, e - H, 0.5);
      if (s80 <= k80) c80++;
      if (s50 <= k50) c50++;
      if (act >= q10 && act <= q90) raw80++;
      sc80L.push({ e: e, s: s80 }); sc50L.push({ e: e, s: s50 });
      var lo = 0; for (k = 0; k < Me; k++) if (q50 + k80 * (dr.sorted[k] + dr.D - q50) < act) lo++;
      pit[Math.min(9, Math.floor(10 * lo / Me))]++;
      var s1 = new Float64Array(Me);
      for (k = 0; k < Me; k++) s1[k] = sim.shS[k * H] + dr.dk[0];
      s1.sort();
      var a1 = Math.log(S.c[i + 1] / S.c[i]), m1 = qSorted(s1, 0.5);
      var sc1 = a1 >= m1 ? (a1 - m1) / Math.max(1e-12, qSorted(s1, 0.9) - m1) : (m1 - a1) / Math.max(1e-12, m1 - qSorted(s1, 0.1));
      if (sc1 <= k80) c1++;
      nc++;
    }
    kappaLive = kappa(sc80L, Infinity, 0.8);
    wf.cov80 = c80 / nc; wf.cov50 = c50 / nc; wf.cov1 = c1 / nc; wf.nCov = nc; wf.pit = pit; wf.raw80 = raw80 / nc; wf.kappa = kappaLive;
    L('pokrytie pásma p10–p90: ' + (100 * wf.cov80).toFixed(1) + ' % (cieľ 80 %, bez kalibrácie ' + (100 * wf.raw80).toFixed(1) + ' %) · šírka ×' + kappaLive.toFixed(2));

    // ---- live direction: every expert retrained on the newest data, mixed with the learned weights
    for (x = 0; x < NX; x++) {
      experts[x].weight = wFinal[x];
      if (!EXP[x].idx) { experts[x].pNow = 0.5; continue; }
      var rwL = rowWeights(aL, bL, cfg.hl), stL = featStats(X, aL, bL, EXP[x].idx), wL = fitLogit(X, y, aL, bL, stL, lam(rwL.sum), wX[x], rwL);
      var pr = predLogit(X, last, stL, wL), cL = fitPlatt(pRawX[x], yE, Ne, H);
      experts[x].pNow = applyPlatt(cL, pr);
      if (x === 0) {
        pRawNow = pr; calL = cL;
        var xNow = rowVec(X, last, stL, new Float64Array(EXP[0].idx.length + 1));
        contrib = EXP[0].idx.map(function (j, q) { return { name: FEAT_NAMES[j], w: wL[q + 1], x: xNow[q + 1], c: wL[q + 1] * xNow[q + 1] }; });
      }
    }
    pNow = 0;
    for (x = 0; x < NX; x++) pNow += experts[x].weight * experts[x].pNow;
    L('učenie: váhy ' + experts.map(function (o) { return o.key + ' ' + Math.round(100 * o.weight) + '%'; }).join(' · ') + ' → P(rast) ' + (100 * pNow).toFixed(1) + ' %');
    } else {
      L('málo histórie na test (' + n + ' sviečok): smer sa neodhaduje, počíta sa len rozpätie', false);
    }

    // ---- live paths (joint SOL+BTC)
    var rnd = mulberry32(seed), poolL = makePool(PS, PB, n - 1500, n);
    var futB = futureBuckets(S.t[last], dt, H);
    var simL = simShocks({ M: Mlive, H: H, PS: PS, PB: PB, pool: poolL, rnd: rnd, hs0: PS.h[n], hb0: PB.h[n], futB: futB, withBtc: true, scale: kappaLive });
    var drL = driftFor(simL, pNow);
    var c0 = S.c[last], cB0 = B.c[last], MH = Mlive * H;
    var C = new Float64Array(MH), Hh = new Float64Array(MH), Lw = new Float64Array(MH), m;
    var termS = new Float64Array(Mlive), termB = new Float64Array(Mlive), maxH = new Float64Array(Mlive), minL = new Float64Array(Mlive);
    for (m = 0; m < Mlive; m++) {
      var prev = c0, mh = -Infinity, ml = Infinity;
      for (k = 0; k < H; k++) {
        var q = m * H + k, rr = simL.shS[q] + drL.dk[k], cc = prev * Math.exp(rr), sg = simL.sgS[q];
        var top = Math.max(prev, cc), bot = Math.min(prev, cc);
        C[q] = cc; Hh[q] = top * Math.exp(simL.uS[q] * sg); Lw[q] = bot * Math.exp(-simL.wS[q] * sg);
        if (Hh[q] > mh) mh = Hh[q];
        if (Lw[q] < ml) ml = Lw[q];
        prev = cc;
      }
      termS[m] = Math.log(prev / c0);
      maxH[m] = mh; minL[m] = ml;
    }
    // BTC paths are centred on a zero median
    var rawB = new Float64Array(Mlive);
    for (m = 0; m < Mlive; m++) { var s2 = 0; for (k = 0; k < H; k++) s2 += simL.shB[m * H + k]; rawB[m] = s2; }
    var medB = quantile(rawB, 0.5);
    for (m = 0; m < Mlive; m++) termB[m] = rawB[m] - medB;

    var QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], bands = {}, pUpStep = new Float64Array(H), col = new Float64Array(Mlive);
    QS.forEach(function (qq) { bands['p' + Math.round(qq * 100)] = new Float64Array(H); });
    var med = new Float64Array(H);
    for (k = 0; k < H; k++) {
      var up = 0;
      for (m = 0; m < Mlive; m++) { col[m] = C[m * H + k]; if (col[m] > c0) up++; }
      pUpStep[k] = up / Mlive;
      var sc = Float64Array.from(col).sort();
      QS.forEach(function (qq) { bands['p' + Math.round(qq * 100)][k] = qSorted(sc, qq); });
      med[k] = bands.p50[k];
    }
    // representative path: closest to the median path
    var best = 0, bestD = Infinity;
    for (m = 0; m < Mlive; m++) {
      var dd = 0;
      for (k = 0; k < H; k++) { var df = Math.log(C[m * H + k] / med[k]); dd += df * df; }
      if (dd < bestD) { bestD = dd; best = m; }
    }
    var rep = { o: new Float64Array(H), h: new Float64Array(H), l: new Float64Array(H), c: new Float64Array(H) };
    for (k = 0; k < H; k++) {
      var qq2 = best * H + k;
      rep.o[k] = k === 0 ? c0 : C[qq2 - 1]; rep.c[k] = C[qq2]; rep.h[k] = Hh[qq2]; rep.l[k] = Lw[qq2];
    }
    var sortedTerm = Float64Array.from(termS).sort();
    var upPaths = 0; for (m = 0; m < Mlive; m++) if (termS[m] > 0) upPaths++;
    var sigmaH = Math.sqrt(sortedTerm.reduce(function (acc, v) { return acc + v * v; }, 0) / Mlive - Math.pow(sortedTerm.reduce(function (a2, v) { return a2 + v; }, 0) / Mlive, 2));

    // touch probabilities
    var sMax = Float64Array.from(maxH).sort(), sMin = Float64Array.from(minL).sort();
    var touch = [0.75, 0.5, 0.25, 0.1].map(function (pq) {
      var upL = qSorted(sMax, 1 - pq), dnL = qSorted(sMin, pq);
      return { p: pq, up: upL, upPct: upL / c0 - 1, dn: dnL, dnPct: dnL / c0 - 1 };
    });

    // BTC conditional scenarios (quintiles of BTC terminal return)
    var order = Array.from({ length: Mlive }, function (_, q) { return q; }).sort(function (a3, b3) { return termB[a3] - termB[b3]; });
    var scen = [];
    for (var g = 0; g < 5; g++) {
      var from = Math.floor(g * Mlive / 5), to = Math.floor((g + 1) * Mlive / 5), bs = [], ssol = [], pu = 0;
      for (var z = from; z < to; z++) { bs.push(termB[order[z]]); ssol.push(termS[order[z]]); if (termS[order[z]] > 0) pu++; }
      scen.push({ btc: Math.exp(quantile(bs, 0.5)) - 1, sol: Math.exp(quantile(ssol, 0.5)) - 1, pUp: pu / (to - from) });
    }

    // ---- regime & BTC statistics
    function er(c, j, Lb) { var s3 = 0; for (var q3 = j - Lb + 1; q3 <= j; q3++) s3 += Math.abs(c[q3] - c[q3 - 1]); return s3 > 0 ? Math.abs(c[j] - c[j - Lb]) / s3 : 0; }
    function corrLag(a4, b4, lag, from4, to4) {
      var ma = 0, mb2 = 0, nn = 0, q4;
      for (q4 = from4; q4 < to4; q4++) { ma += a4[q4]; mb2 += b4[q4 - lag]; nn++; }
      ma /= nn; mb2 /= nn;
      var sab = 0, saa = 0, sbb = 0;
      for (q4 = from4; q4 < to4; q4++) { var da2 = a4[q4] - ma, db2 = b4[q4 - lag] - mb2; sab += da2 * db2; saa += da2 * da2; sbb += db2 * db2; }
      return { corr: sab / Math.sqrt(saa * sbb), beta: sab / sbb };
    }
    var cf = Math.max(2, n - 1000);
    var cc0 = corrLag(PS.r, PB.r, 0, Math.max(2, n - 500), n);
    var leadBS = corrLag(PS.r, PB.r, 1, cf, n).corr, leadSB = corrLag(PB.r, PS.r, 1, cf, n).corr;
    var sigNow = PS.season[futB[0]] * Math.sqrt(PS.h[n]);
    var sigHist = Array.prototype.slice.call(PS.sig, Math.max(1, n - 500), n);
    var volRatio = sigNow / quantile(sigHist, 0.5);
    var LB = Math.min(48, last - 1), erS = er(S.c, last, LB), erB = er(B.c, last, LB);
    var regime = {
      erS: erS, dirS: S.c[last] > S.c[last - LB] ? 1 : -1,
      erB: erB, dirB: B.c[last] > B.c[last - LB] ? 1 : -1,
      volRatio: volRatio, corr: cc0.corr, beta: cc0.beta, leadBS: leadBS, leadSB: leadSB,
      retS48: S.c[last] / S.c[last - LB] - 1, retB48: B.c[last] / B.c[last - LB] - 1, lookback: LB,
      agree: Math.max(upPaths, Mlive - upPaths) / Mlive
    };
    // seasonality peak (weekday hours, UTC)
    var hrs = []; for (k = 0; k < 24; k++) hrs.push({ h: k, s: PS.season[k] });
    hrs.sort(function (a5, b5) { return b5.s - a5.s; });

    L('Monte Carlo ' + Mlive + ' ciest × ' + H + ' sviečok (SOL+BTC spolu)');
    var pst = patternStats(P.ev, S, H, 20, last), adv = patternAdvice(P, pst, S, last);
    L('patterny: ' + P.ev.length + ' v histórii · teraz ' + (adv.active.length ? adv.active.map(function (q) { return PAT[q.key].short; }).join(', ') : 'žiadny'));
    var keep = Math.min(n, 400), from0 = n - keep;
    return {
      meta: { interval: interval, dt: dt, H: H, n: n, M: Mlive, lastT: S.t[last], lastClose: c0, btcClose: cB0, firstT: S.t[0], ms: Date.now() - t0 },
      hist: {
        t: S.t.slice(from0), o: S.o.slice(from0), h: S.h.slice(from0), l: S.l.slice(from0), c: S.c.slice(from0), v: S.v.slice(from0),
        bc: B.c.slice(from0)
      },
      fc: {
        t: Float64Array.from({ length: H }, function (_, q) { return S.t[last] + (q + 1) * dt; }),
        bands: bands, pUpStep: pUpStep, rep: rep,
        samples: Array.from({ length: Math.min(40, Mlive) }, function (_, q) { return C.slice(q * H, q * H + H); }),
        paths: { M: Mlive, H: H, C: C, Hh: Hh, L: Lw, c0: c0 }
      },
      signal: {
        pUp: pNow, pRaw: pRawNow, calSlope: calL.b, median: Math.exp(qSorted(sortedTerm, 0.5)) - 1,
        p10: Math.exp(qSorted(sortedTerm, 0.1)) - 1, p90: Math.exp(qSorted(sortedTerm, 0.9)) - 1,
        p25: Math.exp(qSorted(sortedTerm, 0.25)) - 1, p75: Math.exp(qSorted(sortedTerm, 0.75)) - 1,
        sigmaH: sigmaH, upPaths: upPaths, contrib: contrib, termS: termS
      },
      wf: wf, touch: touch, scen: scen, regime: regime,
      pat: { ev: P.ev, stats: pst, advice: adv, atr: P.A[last] },
      model: {
        garchS: PS.gp, garchB: PB.gp, persistence: PS.gp.alpha + PS.gp.beta,
        halfLife: Math.log(0.5) / Math.log(PS.gp.alpha + PS.gp.beta),
        peakHours: hrs.slice(0, 3).map(function (o) { return o.h; }), sigNext: sigNow,
        experts: experts, features: EXP[0].idx.length, trainRows: lite ? 0 : bL - aL + 1, lambda: lam ? lam(Math.max(1, bL - aL + 1)) : 0, R: R, pool: poolL.idx.length, lite: lite, kappa: kappaLive, cfg: cfg
      },
      log: log
    };
  };

  // Non-overlapping trades in the walk-forward window: enter when p crosses the threshold,
  // hold H bars, pay fee+slippage on both sides.
  E.backtest = function (wf, theta, feeSide) {
    var p = wf.pCal, r = wf.ret, H = wf.H, n = p.length, eq = [1], trades = 0, wins = 0, sum = 0, peak = 1, dd = 0, cur = 1;
    var sh = 0, sl = 0, sigN = 0, sigHit = 0;
    for (var e = wf.mFrom; e < n; e++) {
      if (p[e] >= 0.5 + theta || p[e] <= 0.5 - theta) { sigN++; if ((p[e] > 0.5) === (wf.y[e] === 1)) sigHit++; }
    }
    for (e = wf.mFrom; e < n;) {
      var side = p[e] >= 0.5 + theta ? 1 : (p[e] <= 0.5 - theta ? -1 : 0);
      if (!side) { e++; continue; }
      var net = side * r[e] - 2 * feeSide;
      trades++; if (net > 0) wins++; sum += net;
      if (side > 0) sh++; else sl++;
      cur *= 1 + net; eq.push(cur);
      if (cur > peak) peak = cur;
      dd = Math.max(dd, 1 - cur / peak);
      e += H;
    }
    var hold = 1;
    for (e = wf.mFrom; e < n; e += H) hold *= 1 + r[e];
    var nEff = Math.max(1, sigN / H), sh2 = sigN ? sigHit / sigN : NaN;
    return {
      trades: trades, longs: sh, shorts: sl, win: trades ? wins / trades : NaN, avg: trades ? sum / trades : NaN,
      total: cur - 1, maxDD: dd, equity: eq, hold: hold - 1,
      sigN: sigN, sigHit: sh2, sigCI: sigN ? 1.96 * Math.sqrt(sh2 * (1 - sh2) / nEff) : NaN, sigShare: sigN / Math.max(1, n - wf.mFrom)
    };
  };

  // ---------------------------------------------------------------- decision (shared by the page and the phone notifier)
  E.verdict = function (wf, bt) {
    if (!wf) return { k: 'none', txt: 'neoverené (málo histórie)' };
    var low = bt && isFinite(bt.sigHit) ? bt.sigHit - bt.sigCI : 0;
    if (wf.bss >= 0.01 && low > 0.5) return { k: 'ok', txt: 'model mal v teste výhodu' };
    if (wf.bss > 0) return { k: 'weak', txt: 'výhoda malá, môže byť náhoda' };
    return { k: 'none', txt: 'v teste bez výhody' };
  };
  // LONG / SHORT / wait, how sure, stop-loss and target, and how long to hold: close at the target or
  // the stop, whichever comes first, and at the latest when the forecast horizon ends.
  E.decide = function (res, opt) {
    opt = opt || {};
    var th = opt.theta != null ? opt.theta : 0.05, fee = opt.fee != null ? opt.fee : 0.0007;
    var bt = res.wf ? E.backtest(res.wf, th, fee) : null, v = E.verdict(res.wf, bt), p = res.signal.pUp;
    var raw = p >= 0.5 + th ? 'long' : (p <= 0.5 - th ? 'short' : 'wait');
    var gated = raw !== 'wait' && v.k === 'none', side = gated ? 'wait' : raw, bars = 0;
    if (side !== 'wait') {
      bars = Math.max(1, Math.min(5, Math.round((Math.abs(p - 0.5) - 0.03) / 0.025)));
      if (v.k === 'weak') bars = Math.min(bars, 3);
    }
    var s = res.signal.sigmaH, r5 = function (x) { return Math.max(0.001, Math.round(x * 2000) / 2000); };
    var sl = r5(0.7 * s), tp = r5(1.2 * s), c0 = res.meta.lastClose, dt = res.meta.dt, H = res.meta.H, d = raw === 'short' ? -1 : 1;
    var plans = E.tradePlan(res.fc.paths, tp, sl, fee), t0 = res.meta.lastT + dt;
    return { p: p, raw: raw, side: side, gated: gated, bars: bars, v: v, bt: bt, sl: sl, tp: tp, entry: c0,
      slPx: c0 * (1 - d * sl), tpPx: c0 * (1 + d * tp), t0: t0, tEnd: t0 + H * dt, H: H, dt: dt,
      plan: raw === 'short' ? plans.short : plans.long, plans: plans };
  };
  // Where an open trade stands: closes at the stop or target touched first (stop first when both fall
  // inside one candle), at the latest at tEnd; 'flip' when the model now points the other way.
  E.positionStatus = function (pos, candles, now, nowPx, modelSide) {
    var d = pos.side === 'short' ? -1 : 1, hit = null, best = 0, worst = 0, i;
    for (i = 0; i < candles.t.length; i++) {
      if (candles.t[i] < pos.t0) continue;
      var hi = candles.h[i], lo = candles.l[i];
      best = Math.max(best, d > 0 ? hi / pos.entry - 1 : 1 - lo / pos.entry);
      worst = Math.min(worst, d > 0 ? lo / pos.entry - 1 : 1 - hi / pos.entry);
      if (d > 0 ? lo <= pos.slPx : hi >= pos.slPx) { hit = { k: 'sl', t: candles.t[i] }; break; }
      if (d > 0 ? hi >= pos.tpPx : lo <= pos.tpPx) { hit = { k: 'tp', t: candles.t[i] }; break; }
    }
    if (!hit && isFinite(nowPx)) {
      if (d > 0 ? nowPx <= pos.slPx : nowPx >= pos.slPx) hit = { k: 'sl', t: now };
      else if (d > 0 ? nowPx >= pos.tpPx : nowPx <= pos.tpPx) hit = { k: 'tp', t: now };
    }
    var k = hit ? hit.k : (now >= pos.tEnd ? 'time' : (modelSide && modelSide !== 'wait' && modelSide !== pos.side ? 'flip' : 'hold'));
    var ret = isFinite(nowPx) ? d * (nowPx / pos.entry - 1) : NaN;
    return { k: k, at: hit ? hit.t : null, ret: ret, best: best, worst: worst, left: pos.tEnd - now,
      beHint: k === 'hold' && best >= 0.6 * pos.tp };
  };

  // First-touch TP/SL on the simulated candle paths (SL assumed first if both inside one candle).
  E.tradePlan = function (paths, tp, sl, feeSide) {
    var M = paths.M, H = paths.H, C = paths.C, Hh = paths.Hh, L = paths.L, c0 = paths.c0;
    function med(a) { if (!a.length) return NaN; a.sort(function (x, y) { return x - y; }); return a[Math.floor(a.length / 2)]; }
    function side(dir) {
      var nTP = 0, nSL = 0, nT = 0, acc = 0, rets = new Float64Array(M), tT = [], tS = [];
      for (var m = 0; m < M; m++) {
        var res = null;
        for (var k = 0; k < H; k++) {
          var q = m * H + k, hi = Hh[q], lo = L[q];
          var slHit = dir > 0 ? lo <= c0 * (1 - sl) : hi >= c0 * (1 + sl);
          var tpHit = dir > 0 ? hi >= c0 * (1 + tp) : lo <= c0 * (1 - tp);
          if (slHit) { res = -sl; nSL++; tS.push(k + 1); break; }
          if (tpHit) { res = tp; nTP++; tT.push(k + 1); break; }
        }
        if (res === null) { res = dir * (C[m * H + H - 1] / c0 - 1); nT++; }
        rets[m] = res - 2 * feeSide; acc += rets[m];
      }
      // tTP / tSL: typical number of candles until the target / stop is touched (when it is touched first)
      return { pTP: nTP / M, pSL: nSL / M, pTime: nT / M, ev: acc / M, tTP: med(tT), tSL: med(tS) };
    }
    return { long: side(1), short: side(-1) };
  };

  // ---------------------------------------------------------------- synthetic data (demo + tests)
  E.genSynthetic = function (o) {
    o = o || {};
    var n = o.n || 3000, dt = INTERVAL_MS[o.interval || '15m'], rnd = mulberry32(o.seed || 7), g = makeGauss(rnd);
    var end = Math.floor((o.endTime || Date.now()) / dt) * dt - dt;
    var lead = o.leadLag != null ? o.leadLag : 0.06, mom = o.momentum || 0;
    var sc = Math.sqrt(dt / 900000), baseB = 0.0024 * sc, baseI = 0.0026 * sc, betaS = 1.25;
    function prof(t) {
      var d = new Date(t), h = d.getUTCHours() + d.getUTCMinutes() / 60, wd = d.getUTCDay();
      var f = 1 + 0.45 * Math.exp(-Math.pow(h - 14.5, 2) / 6) + 0.15 * Math.exp(-Math.pow(h - 1, 2) / 4) - 0.22 * Math.exp(-Math.pow(h - 5, 2) / 5);
      return f * ((wd === 0 || wd === 6) ? 0.72 : 1);
    }
    function tdraw() { var nu = 5, c2 = 0; for (var q = 0; q < nu; q++) { var gg = g(); c2 += gg * gg; } return g() / Math.sqrt(c2 / nu) * Math.sqrt((nu - 2) / nu); }
    function series() { return { t: [], o: [], h: [], l: [], c: [], v: [], tb: [] }; }
    var Sx = series(), Bx = series(), cS = o.sol0 || 180, cB = o.btc0 || 100000, hB = 1, hI = 1, prevZB = 0, prevRS = 0;
    for (var i = 0; i < n; i++) {
      var t = end - (n - 1 - i) * dt, f = prof(t);
      var zB = tdraw(), zI = tdraw();
      var sigB = f * baseB * Math.sqrt(hB);
      var rB = sigB * zB;
      var sigS = Math.sqrt(betaS * betaS * baseB * baseB * hB + baseI * baseI * hI) * f;
      var rS = betaS * rB + f * baseI * Math.sqrt(hI) * zI + lead * prevZB * sigS + mom * prevRS;
      hB = 0.03 + 0.07 * zB * zB * hB + 0.90 * hB; hI = 0.04 + 0.08 * zI * zI * hI + 0.88 * hI;
      prevZB = zB; prevRS = rS;
      [[Sx, rS, cS, sigS], [Bx, rB, cB, sigB]].forEach(function (a) {
        var s = a[0], r = a[1], c = a[2], sg = a[3], oo = c, cc = c * Math.exp(r);
        s.t.push(t); s.o.push(oo); s.c.push(cc);
        s.h.push(Math.max(oo, cc) * Math.exp(Math.abs(g()) * 0.5 * sg));
        s.l.push(Math.min(oo, cc) * Math.exp(-Math.abs(g()) * 0.5 * sg));
        var vol = 1000 * f * Math.exp(0.45 * g()) * (1 + 1.5 * Math.abs(r) / sg);
        s.v.push(vol);
        s.tb.push(vol * clip(0.5 + 0.16 * Math.tanh(r / sg) + 0.04 * g(), 0.05, 0.95));
      });
      cS = Sx.c[i]; cB = Bx.c[i];
    }
    return { sol: Sx, btc: Bx };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = E;
  else root.SolEngine = E;
})(typeof self !== 'undefined' ? self : this);

// =====================================================================================
// SOL predikcia – notifikácie na mobil (GitHub Actions + ntfy)
// Raz za ~15 minút stiahne sviečky SOL a BTC, prepočíta ten istý model ako appka a pri novom
// signáli LONG/SHORT alebo pri čase na predaj pošle notifikáciu do appky ntfy na mobile.
// Nastavenia sú hneď nižšie. Tajný názov kanála ntfy je v GitHub Secrets ako NTFY_TOPIC.
// =====================================================================================
const NASTAVENIA = {
  interval: '15m',            // sviečky: '5m', '15m', '1h' alebo '4h'
  dopredu: 24,                // na koľko sviečok dopredu (24 × 15 min = 6 hodín)
  casovaZona: 'Europe/London',// čas v notifikáciách
  appka: 'https://claude.ai/artifact/Anygy256N1qm7YM8GVmmTu',
  ntfyServer: 'https://ntfy.sh'
};
// =====================================================================================

const fs = require('fs');
const SE = module.exports;
const THETA = 0.05, FEE = 0.0007, NFETCH = 5000, STATE_FILE = 'state.json';
const IV_MS = { '5m': 300000, '15m': 900000, '1h': 3600000, '4h': 14400000 };
const env = process.env;
const TOPIC = (env.NTFY_TOPIC || '').trim();
const TEST = env.TEST === 'true';
const NTFY = env.NTFY_SERVER || NASTAVENIA.ntfyServer;

function log() { console.log.apply(console, arguments); }
function fp(x) { return isFinite(x) ? x.toLocaleString('sk-SK', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '–'; }
function pc(x) { return (x > 0 ? '+' : (x < 0 ? '−' : '')) + Math.abs(x * 100).toLocaleString('sk-SK', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + ' %'; }
function hm(t) { return new Intl.DateTimeFormat('sk-SK', { timeZone: NASTAVENIA.casovaZona, hour: '2-digit', minute: '2-digit' }).format(new Date(t)); }
function readJson(f, d) { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (e) { return d; } }
async function getJSON(u) {
  const ctl = new AbortController(), to = setTimeout(() => ctl.abort(), 20000);
  try {
    const r = await fetch(u, { signal: ctl.signal, headers: { 'User-Agent': 'sol-predikcia-notifier' } });
    if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + u.split('?')[0]);
    return await r.json();
  } finally { clearTimeout(to); }
}

// ---- exchanges (the first one that answers wins)
const B = (env.BASE_BINANCE || 'https://data-api.binance.vision');
const binance = {
  name: 'Binance', max: 1000,
  page: async (sym, iv, lim, end) => (await getJSON(B + '/api/v3/klines?symbol=' + sym + '&interval=' + iv + '&limit=' + lim + (end ? '&endTime=' + end : '')))
    .map(r => ({ t: +r[0], o: +r[1], h: +r[2], l: +r[3], c: +r[4], v: +r[5], tb: +r[9] }))
};
const CDC_TF = { '5m': 'M5', '15m': 'M15', '1h': 'H1', '4h': 'H4' };
const cryptocom = {
  name: 'Crypto.com', max: 300,
  page: async (sym, iv, lim, end) => {
    const inst = sym.replace('USDT', '_USDT');
    const j = await getJSON((env.BASE_CDC || 'https://api.crypto.com') + '/exchange/v1/public/get-candlestick?instrument_name=' + inst + '&timeframe=' + CDC_TF[iv] + '&count=' + lim + (end ? '&end_ts=' + (end + 1) : ''));
    const d = j && j.result && j.result.data;
    if (!Array.isArray(d)) throw new Error('Crypto.com: zlá odpoveď');
    return d.map(r => ({ t: +r.t, o: +r.o, h: +r.h, l: +r.l, c: +r.c, v: +r.v, tb: NaN })).sort((a, b) => a.t - b.t);
  }
};
const CB_GRAN = { '5m': 300, '15m': 900, '1h': 3600 }, CB_PROD = { SOLUSDT: 'SOL-USD', BTCUSDT: 'BTC-USD' };
const coinbase = {
  name: 'Coinbase', max: 300,
  page: async (sym, iv, lim, end) => {
    const g = CB_GRAN[iv];
    if (!g) throw new Error('Coinbase nemá ' + iv);
    const e = end ? end + 1 : Date.now(), s = e - lim * g * 1000;
    const j = await getJSON((env.BASE_CB || 'https://api.exchange.coinbase.com') + '/products/' + CB_PROD[sym] + '/candles?granularity=' + g + '&start=' + new Date(s).toISOString() + '&end=' + new Date(e).toISOString());
    if (!Array.isArray(j)) throw new Error('Coinbase: zlá odpoveď');
    return j.map(r => ({ t: r[0] * 1000, o: +r[3], h: +r[2], l: +r[1], c: +r[4], v: +r[5], tb: NaN })).sort((a, b) => a.t - b.t);
  }
};
const BYBIT_IV = { '5m': '5', '15m': '15', '1h': '60', '4h': '240' };
const bybit = {
  name: 'Bybit', max: 1000,
  page: async (sym, iv, lim, end) => {
    const j = await getJSON((env.BASE_BYBIT || 'https://api.bybit.com') + '/v5/market/kline?category=spot&symbol=' + sym + '&interval=' + BYBIT_IV[iv] + '&limit=' + lim + (end ? '&end=' + end : ''));
    if (!j || !j.result || !Array.isArray(j.result.list)) throw new Error('Bybit: zlá odpoveď');
    return j.result.list.map(r => ({ t: +r[0], o: +r[1], h: +r[2], l: +r[3], c: +r[4], v: +r[5], tb: NaN })).reverse();
  }
};
const SOURCES = [binance, cryptocom, coinbase, bybit];

async function history(src, sym, iv, n) {
  let rows = [], end = null, guard = 0;
  while (rows.length < n && guard++ < 40) {
    const page = await src.page(sym, iv, src.max, end);
    if (!page.length) break;
    const first = rows.length ? rows[0].t : Infinity;
    const older = page.filter(r => r.t < first);
    if (!older.length) break;
    rows = older.concat(rows);
    end = older[0].t - 1;
    if (page.length < src.max * 0.5 && src !== coinbase) break;
  }
  const seen = new Set(), out = [];
  rows.forEach(r => { if (!seen.has(r.t) && r.c > 0) { seen.add(r.t); out.push(r); } });
  out.sort((a, b) => a.t - b.t);
  return out.slice(-n);
}
function toSeries(rows, dt) { // closed candles only
  const now = Date.now(), s = { t: [], o: [], h: [], l: [], c: [], v: [], tb: [] };
  rows.forEach(r => { if (r.t + dt <= now) { s.t.push(r.t); s.o.push(r.o); s.h.push(r.h); s.l.push(r.l); s.c.push(r.c); s.v.push(r.v); s.tb.push(r.tb); } });
  return s;
}
async function loadData(iv) {
  const errs = [], dt = IV_MS[iv];
  for (const src of SOURCES) {
    try {
      const [a, b] = await Promise.all([history(src, 'SOLUSDT', iv, NFETCH), history(src, 'BTCUSDT', iv, NFETCH)]);
      const sol = toSeries(a, dt), btc = toSeries(b, dt);
      if (sol.t.length < 1000 || btc.t.length < 1000) throw new Error(src.name + ': málo sviečok (' + sol.t.length + ')');
      if (Date.now() - sol.t[sol.t.length - 1] > 3 * dt + 120000) throw new Error(src.name + ': staré dáta');
      const last = a[a.length - 1];
      return { src: src.name, sol, btc, nowPx: last ? last.c : sol.c[sol.c.length - 1] };
    } catch (e) { errs.push(e.message || String(e)); log('zdroj nevyšiel:', e.message || e); }
  }
  throw new Error('Žiadna burza neodpovedala: ' + errs.join(' | '));
}

async function ntfy(msg) {
  if (!TOPIC) { log('NTFY_TOPIC nie je nastavený, notifikácia by bola:', JSON.stringify(msg)); return false; }
  const body = Object.assign({ topic: TOPIC, click: NASTAVENIA.appka }, msg);
  const r = await fetch(NTFY, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  log('ntfy', r.status, msg.title);
  return r.ok;
}

async function main() {
  const iv = NASTAVENIA.interval, H = NASTAVENIA.dopredu, dt = IV_MS[iv];
  const st = readJson(STATE_FILE, {});
  const data = await loadData(iv);
  const lastT = data.sol.t[data.sol.t.length - 1];
  const res = SE.run(data.sol, data.btc, { interval: iv, H: H, M: 400, seed: Math.floor(lastT / 1000) % 2147483647 });
  let d = SE.decide(res, { theta: THETA, fee: FEE });
  if (env.SIM_SIDE) d = Object.assign({}, d, { side: env.SIM_SIDE, raw: env.SIM_SIDE, bars: 3 });   // only for tests
  const now = Date.now(), px = data.nowPx, msgs = [];
  log('zdroj', data.src, '· sviečok', res.meta.n, '· šanca rastu', (100 * d.p).toFixed(1) + ' %', '· overenie', d.v.k, '· signál', d.side);

  // 1) the signal we told the phone about earlier: target, stop, time, or the model turned
  if (st.active) {
    const a = st.active, candles = { t: [], h: [], l: [] };
    data.sol.t.forEach((t, i) => { if (t >= a.t0) { candles.t.push(t); candles.h.push(data.sol.h[i]); candles.l.push(data.sol.l[i]); } });
    const s = SE.positionStatus(a, candles, now, px, d.side);
    if (s.k !== 'hold') {
      const close = a.side === 'short' ? 'UZAVRIEŤ SHORT' : 'PREDAŤ';
      const why = { tp: 'cieľ ' + fp(a.tpPx) + ' dosiahnutý', sl: 'stop-loss ' + fp(a.slPx), time: 'čas vypršal (' + hm(a.tEnd) + ')', flip: 'model sa otočil na ' + d.side.toUpperCase() }[s.k];
      msgs.push({ title: 'SOL: ' + close + ' – ' + why, message: 'Signál ' + a.side.toUpperCase() + ' od ' + hm(a.t0) + ' za ' + fp(a.entry) + '. Teraz ' + fp(px) + ' (' + pc(s.ret) + '). Ak máš pozíciu otvorenú, je čas ju uzavrieť.',
        priority: s.k === 'sl' ? 5 : 4, tags: [s.k === 'tp' ? 'white_check_mark' : (s.k === 'sl' ? 'warning' : 'hourglass')] });
      st.history = (st.history || []).concat([{ side: a.side, t0: a.t0, entry: a.entry, t1: now, exit: px, why: s.k, ret: s.ret }]).slice(-50);
      st.active = null;
      st.cooldownUntil = now + dt;   // a fresh signal of the same side waits for the next run
    }
  }
  // 2) a new LONG / SHORT
  if (d.side !== 'wait' && !st.active && !(st.cooldownUntil && now < st.cooldownUntil && st.lastSide === d.side)) {
    const dir = d.side === 'long' ? 1 : -1, tpPx = px * (1 + dir * d.tp), slPx = px * (1 - dir * d.sl), tEnd = now + H * dt;
    const istota = ['', 'veľmi nízka', 'nízka', 'stredná', 'vyššia', 'vysoká'][d.bars] || '';
    msgs.push({ title: 'SOL: ' + d.side.toUpperCase() + ' za ' + fp(px), message: 'Cieľ ' + fp(tpPx) + ' (' + pc(dir * d.tp) + ') · stop ' + fp(slPx) + ' (' + pc(-dir * d.sl) + ') · drž najdlhšie do ' + hm(tEnd) +
      '. Šanca ' + (dir > 0 ? 'rastu ' : 'poklesu ') + Math.round(100 * (dir > 0 ? d.p : 1 - d.p)) + ' %, istota ' + istota + '. Over v appke pred obchodom.',
      priority: 4, tags: [dir > 0 ? 'chart_with_upwards_trend' : 'chart_with_downwards_trend'] });
    st.active = { side: d.side, entry: px, t0: now, tp: d.tp, sl: d.sl, tpPx: tpPx, slPx: slPx, tEnd: tEnd };
  }
  // 3) manual test run from GitHub
  if (TEST) msgs.push({ title: 'SOL predikcia: test notifikácií ✓', message: 'Funguje to. Zdroj ' + data.src + ', posledná sviečka ' + hm(lastT + dt) + ', cena ' + fp(px) + ', model teraz: ' +
    (d.side === 'wait' ? 'ČAKAŤ' : d.side.toUpperCase()) + ' (šanca rastu ' + Math.round(100 * d.p) + ' %).', priority: 3, tags: ['bell'] });

  for (const m of msgs) await ntfy(m);
  st.lastSide = d.side;
  st.week = new Date(now).toISOString().slice(0, 10).replace(/-\d\d$/, '') + '-w' + Math.floor(new Date(now).getUTCDate() / 7);   // one small change a week keeps GitHub from pausing the schedule
  st.src = data.src;
  fs.writeFileSync(STATE_FILE, JSON.stringify(st, null, 1) + '\n');
}

main().catch(async e => {
  log('CHYBA:', e && e.stack || e);
  if (TEST) { try { await ntfy({ title: 'SOL predikcia: test zlyhal', message: String(e && e.message || e).slice(0, 300), priority: 3, tags: ['x'] }); } catch (x) { } }
  process.exitCode = 1;
});
