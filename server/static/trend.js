/* Historical trend page — server/static/trend.js
   Time-series Plotly chart with six metric traces, regression/improvement
   markers, dataset_version filter, single-run and empty-state handling. */
(function () {
    'use strict';

    var R = window.RagBench;
    if (!R) {
        console.error('RagBench common.js not loaded');
        return;
    }

    // ── Metric definitions ──
    var METRICS = [
        { key: 'composite_score',   label: 'Composite Score',   higherBetter: true  },
        { key: 'hit_at_5',          label: 'Hit@5',             higherBetter: true  },
        { key: 'symbol_hit_at_5',   label: 'Symbol Hit@5',      higherBetter: true  },
        { key: 'mrr',               label: 'MRR',               higherBetter: true  },
        { key: 'latency_p50_ms',    label: 'Latency P50 (ms)',  higherBetter: false },
        { key: 'avg_response_tokens', label: 'Avg Response Tokens', higherBetter: false }
    ];

    var PALETTE = R.palette;  // 6 hex codes

    // Marker symbols
    var SYM_NORMAL = 'circle';
    var SYM_REGRESSION = 'x-thin';
    var SYM_IMPROVEMENT = 'triangle-up';

    // Colors
    var COLOR_REGRESSION = '#f85149';   // bad / red
    var COLOR_IMPROVEMENT = '#3fb950';  // good / green

    var REGRESSION_THRESHOLD = 0.05;  // 5%

    // ── Parsing ──
    function getServerName() {
        // URL pattern: /server/{server_name}/trend
        var path = window.location.pathname;
        var m = path.match(/^\/server\/(.+)\/trend$/);
        return m ? decodeURIComponent(m[1]) : null;
    }

    // ── Compute per-point marker metadata ──
    // For each metric trace, compute arrays: markerColor[], markerSymbol[], annotationText[]
    // based on the >5% direction-aware rule.
    function computeMarkers(entries) {
        var n = entries.length;
        // Initialize arrays for each metric
        var markers = {};
        METRICS.forEach(function (m) {
            markers[m.key] = {
                color: new Array(n),
                symbol: new Array(n),
                annotations: []  // { x, y, text, color }
            };
        });

        METRICS.forEach(function (metric) {
            var mk = markers[metric.key];
            var higherBetter = metric.higherBetter;

            for (var i = 0; i < n; i++) {
                if (i === 0) {
                    // First entry: no regression/improvement, normal marker
                    mk.color[i] = PALETTE[METRICS.indexOf(metric)];
                    mk.symbol[i] = SYM_NORMAL;
                } else {
                    var prevVal = entries[i - 1][metric.key];
                    var currVal = entries[i][metric.key];
                    var prev = prevVal != null ? Number(prevVal) : 0;
                    var curr = currVal != null ? Number(currVal) : 0;

                    var isRegression = false;
                    var isImprovement = false;

                    if (prev !== 0) {
                        if (higherBetter) {
                            // Higher-is-better
                            // Regression: prev dropped by >5%  → (prev - curr) / prev > 0.05
                            // Improvement: curr improved by >5% → (curr - prev) / prev > 0.05
                            isRegression = (prev - curr) / prev > REGRESSION_THRESHOLD;
                            isImprovement = (curr - prev) / prev > REGRESSION_THRESHOLD;
                        } else {
                            // Lower-is-better
                            // Regression: curr increased by >5%  → (curr - prev) / prev > 0.05
                            // Improvement: curr decreased by >5% → (prev - curr) / prev > 0.05
                            isRegression = (curr - prev) / prev > REGRESSION_THRESHOLD;
                            isImprovement = (prev - curr) / prev > REGRESSION_THRESHOLD;
                        }
                    }

                    if (isRegression) {
                        mk.color[i] = COLOR_REGRESSION;
                        mk.symbol[i] = SYM_REGRESSION;
                        mk.annotations.push({
                            x: entries[i].submitted_at,
                            y: curr,
                            text: '&#9660;',  // down triangle
                            color: COLOR_REGRESSION
                        });
                    } else if (isImprovement) {
                        mk.color[i] = COLOR_IMPROVEMENT;
                        mk.symbol[i] = SYM_IMPROVEMENT;
                        mk.annotations.push({
                            x: entries[i].submitted_at,
                            y: curr,
                            text: '&#9650;',  // up triangle
                            color: COLOR_IMPROVEMENT
                        });
                    } else {
                        mk.color[i] = PALETTE[METRICS.indexOf(metric)];
                        mk.symbol[i] = SYM_NORMAL;
                    }
                }
            }
        });

        return markers;
    }

    // ── Build Plotly traces ──
    function buildTraces(entries, markers) {
        var traces = [];
        var allAnnotations = [];

        METRICS.forEach(function (metric, idx) {
            var mk = markers[metric.key];
            var xVals = entries.map(function (e) { return e.submitted_at; });
            var yVals = entries.map(function (e) { return e[metric.key]; });
            var runIds = entries.map(function (e) { return e.run_id; });

            traces.push({
                x: xVals,
                y: yVals,
                type: 'scatter',
                mode: 'lines+markers',
                name: metric.label,
                line: {
                    color: PALETTE[idx],
                    width: 2
                },
                marker: {
                    color: mk.color,
                    symbol: mk.symbol,
                    size: 8,
                    line: {
                        color: mk.color.map(function (c) {
                            // Give regression/improvement markers a contrasting border
                            return (c === COLOR_REGRESSION || c === COLOR_IMPROVEMENT) ? '#c9d1d9' : c;
                        }),
                        width: 1
                    }
                },
                customdata: runIds,
                hovertemplate: (
                    '<b>' + R.esc(metric.label) + '</b>: %{y:.4g}<br>' +
                    'Submitted: %{x}<br>' +
                    'Run: %{customdata}<extra></extra>'
                )
            });

            // Collect annotations for this trace (regression/improvement markers)
            mk.annotations.forEach(function (ann) {
                allAnnotations.push({
                    x: ann.x,
                    y: ann.y,
                    text: ann.text,
                    showarrow: true,
                    arrowhead: 1,
                    arrowsize: 1,
                    arrowwidth: 1,
                    arrowcolor: ann.color,
                    font: { color: ann.color, size: 14 },
                    ax: 0,
                    ay: -25,
                    xref: 'x',
                    yref: 'y'
                });
            });
        });

        return { traces: traces, annotations: allAnnotations };
    }

    // ── Filter traces by dataset_version ──
    function filterTraces(entries, markers, version) {
        // Create filtered entries array
        var filtered = version === 'all'
            ? entries
            : entries.filter(function (e) { return e.dataset_version === version; });

        // Recompute markers only for the filtered set
        var filteredMarkers = computeMarkers(filtered);

        return buildTraces(filtered, filteredMarkers);
    }

    // ── Main render ──
    function render(entries, serverName) {
        var versionSet = {};
        entries.forEach(function (e) {
            if (e.dataset_version) versionSet[e.dataset_version] = true;
        });
        var versions = Object.keys(versionSet).sort();

        // Show/hide version filter
        var filterSelect = document.getElementById('versionFilter');
        if (versions.length > 1) {
            filterSelect.removeAttribute('hidden');
            // Populate options
            filterSelect.innerHTML = '<option value="all">All</option>' +
                versions.map(function (v) {
                    return '<option value="' + R.escAttr(v) + '">' + R.esc(v) + '</option>';
                }).join('');
        } else {
            filterSelect.setAttribute('hidden', '');
        }

        // Compute markers and build initial traces
        var markers = computeMarkers(entries);
        var built = buildTraces(entries, markers);

        var layout = R.plotlyLayout({
            title: {
                text: 'Metrics over Time — ' + R.esc(serverName),
                font: { color: '#c9d1d9', size: 16 }
            },
            xaxis: {
                title: { text: 'Submission Date', font: { color: '#8b949e' } },
                type: 'date',
                gridcolor: '#30363d',
                zerolinecolor: '#30363d'
            },
            yaxis: {
                title: { text: 'Value', font: { color: '#8b949e' } },
                gridcolor: '#30363d',
                zerolinecolor: '#30363d'
            },
            annotations: built.annotations,
            hovermode: 'closest',
            legend: {
                font: { color: '#c9d1d9' },
                orientation: 'h',
                y: -0.2
            },
            margin: { l: 70, r: 30, t: 60, b: 80 }
        });

        var config = {
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            responsive: true
        };

        Plotly.newPlot('trendChart', built.traces, layout, config);

        // ── Click handler: navigate to /run/{run_id}/report ──
        var trendDiv = document.getElementById('trendChart');
        trendDiv.on('plotly_click', function (data) {
            if (!data || !data.points || data.points.length === 0) return;
            var pt = data.points[0];
            if (pt.customdata) {
                window.location.href = '/run/' + encodeURIComponent(pt.customdata) + '/report';
            }
        });

        // ── Version filter handler ──
        filterSelect.addEventListener('change', function () {
            var sel = filterSelect.value;
            var filtered = filterTraces(entries, markers, sel);

            var newLayout = R.plotlyLayout({
                title: {
                    text: 'Metrics over Time — ' + R.esc(serverName),
                    font: { color: '#c9d1d9', size: 16 }
                },
                xaxis: {
                    title: { text: 'Submission Date', font: { color: '#8b949e' } },
                    type: 'date',
                    gridcolor: '#30363d',
                    zerolinecolor: '#30363d'
                },
                yaxis: {
                    title: { text: 'Value', font: { color: '#8b949e' } },
                    gridcolor: '#30363d',
                    zerolinecolor: '#30363d'
                },
                annotations: filtered.annotations,
                hovermode: 'closest',
                legend: {
                    font: { color: '#c9d1d9' },
                    orientation: 'h',
                    y: -0.2
                },
                margin: { l: 70, r: 30, t: 60, b: 80 }
            });

            Plotly.react('trendChart', filtered.traces, newLayout, config);
        });

        // Show content, hide loading
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('errorState').style.display = 'none';
        document.getElementById('trendContent').style.display = 'block';

        // Update header info
        document.getElementById('trendTitle').textContent = serverName;
        var meta = entries.length + ' run' + (entries.length !== 1 ? 's' : '');
        if (versions.length > 0) {
            meta += ' · ' + versions.length + ' dataset version' + (versions.length !== 1 ? 's' : '');
        }
        document.getElementById('trendMeta').textContent = meta;
    }

    function showError(msg) {
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('trendContent').style.display = 'none';
        var errEl = document.getElementById('errorState');
        errEl.style.display = 'block';
        if (msg) {
            document.getElementById('errorDetail').textContent = msg;
        }
    }

    // ── Bootstrap ──
    async function init() {
        var serverName = getServerName();
        if (!serverName) {
            showError('Could not determine server name from URL.');
            return;
        }

        document.title = serverName + ' Trend — rag-bench';

        var history;
        try {
            history = await R.fetchJson('/api/server/' + encodeURIComponent(serverName) + '/history');
        } catch (err) {
            console.error('Failed to fetch history:', err);
            showError('Failed to load trend data. The server may be unavailable.');
            return;
        }

        if (!Array.isArray(history) || history.length === 0) {
            showError('There are no benchmark runs for this server yet.');
            return;
        }

        render(history, serverName);
    }

    // Wait for Plotly to be available, then init
    if (typeof Plotly !== 'undefined') {
        init();
    } else {
        window.addEventListener('load', init);
    }
})();
