/* rag-bench shared frontend utilities — exposed on window.RagBench */
(function () {
    'use strict';

    var palette = [
        '#58a6ff',  /* accent */
        '#3fb950',  /* good */
        '#d29922',  /* mid */
        '#f85149',  /* bad */
        '#bc8cff',
        '#79c0ff'
    ];

    /**
     * Build a Plotly layout object with the dark-theme defaults.
     * Callers pass overrides that are shallow-merged.
     */
    function plotlyLayout(overrides) {
        var base = {
            paper_bgcolor: '#0d1117',
            plot_bgcolor: '#0d1117',
            font: {
                color: '#c9d1d9',
                size: 12
            },
            xaxis: {
                gridcolor: '#30363d',
                zerolinecolor: '#30363d'
            },
            yaxis: {
                gridcolor: '#30363d',
                zerolinecolor: '#30363d'
            },
            legend: {
                font: { color: '#c9d1d9' }
            },
            margin: { l: 60, r: 30, t: 50, b: 60 }
        };
        if (!overrides) return base;
        return mergeDeep(base, overrides);
    }

    function mergeDeep(target, source) {
        var out = {};
        var key;
        for (key in target) {
            if (target.hasOwnProperty(key)) out[key] = target[key];
        }
        for (key in source) {
            if (source.hasOwnProperty(key)) {
                if (isObject(source[key]) && isObject(target[key])) {
                    out[key] = mergeDeep(target[key], source[key]);
                } else {
                    out[key] = source[key];
                }
            }
        }
        return out;
    }

    function isObject(v) {
        return v !== null && typeof v === 'object' && !Array.isArray(v);
    }

    /**
     * Fetch JSON from a URL. Throws on non-ok responses.
     */
    async function fetchJson(url) {
        var res = await fetch(url);
        if (!res.ok) {
            throw new Error('HTTP ' + res.status + ' for ' + url);
        }
        return res.json();
    }

    /** HTML-escape a string for safe innerHTML insertion. */
    function esc(s) {
        if (s == null) return '';
        var d = document.createElement('div');
        d.textContent = String(s);
        return d.innerHTML;
    }

    /** Escape a string for use in an HTML attribute value. */
    function escAttr(s) {
        if (s == null) return '';
        return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    /** Format a token count with locale-aware grouping. */
    function fmtTokens(v) {
        if (v == null || v === 0) return '0';
        return Math.round(v).toLocaleString();
    }

    /** Format a proportion (0–1) as a percentage string. */
    function pct(v) {
        return (v * 100).toFixed(1) + '%';
    }

    /** Return a CSS class name for the given composite score. */
    function scoreClass(s) {
        return s >= 0.7 ? 'good' : s >= 0.4 ? 'mid' : 'bad';
    }

    window.RagBench = {
        palette: palette,
        plotlyLayout: plotlyLayout,
        fetchJson: fetchJson,
        esc: esc,
        escAttr: escAttr,
        fmtTokens: fmtTokens,
        pct: pct,
        scoreClass: scoreClass
    };
})();
