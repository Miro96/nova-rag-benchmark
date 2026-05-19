/* rag-bench multi-run comparison page — /compare?run_ids=... */
(function () {
    'use strict';

    var R = window.RagBench;
    var palette = R.palette;
    var plotlyLayout = R.plotlyLayout;

    // ── DOM refs ──
    var loadingEl = document.getElementById('loadingState');
    var errorEl = document.getElementById('errorState');
    var errorDetailEl = document.getElementById('errorDetail');
    var contentEl = document.getElementById('compareContent');
    var compareTitleEl = document.getElementById('compareTitle');
    var compareMetaEl = document.getElementById('compareMeta');

    // ── Parse run_ids from URL ──
    var params = new URLSearchParams(window.location.search);
    var runIdsRaw = params.get('run_ids') || '';
    var runIds = runIdsRaw.split(',').map(function (s) { return s.trim(); }).filter(Boolean);

    // ── State ──
    var compareData = null;       // /api/compare response
    var perRunStats = {};         // run_id → stats from /api/run/{id}/stats

    // Plotly config
    var PLOTLY_CONFIG = {
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        responsive: true
    };

    // ── Helper: safe number ──
    function safeNum(v, d) {
        d = d === undefined ? 0 : d;
        if (v === null || v === undefined || isNaN(v)) return d;
        return v;
    }

    // ── Metrics used for comparison ──
    var COMPARE_METRICS = ['hit_at_5', 'mrr', 'query_latency_p50_ms'];
    var METRIC_LABELS = {
        hit_at_5: 'Hit@5',
        mrr: 'MRR',
        query_latency_p50_ms: 'Latency p50 (ms)'
    };

    // ── Error state ──
    function showError(msg) {
        loadingEl.style.display = 'none';
        errorEl.style.display = 'block';
        contentEl.style.display = 'none';
        errorDetailEl.innerHTML = msg;
    }

    // ── Number formatting ──
    function fmtNum(v) {
        if (v === null || v === undefined) return '—';
        if (typeof v === 'number') {
            if (v === 0) return '0.0000';
            if (Math.abs(v) < 0.001) return v.toExponential(3);
            return v.toFixed(4);
        }
        return String(v);
    }

    function fmtPval(v) {
        if (v === null || v === undefined) return '—';
        if (typeof v !== 'number') return String(v);
        if (v < 0.001) return v.toExponential(3);
        return v.toFixed(4);
    }

    function fmtCI(lo, hi) {
        return '[' + fmtNum(lo) + ', ' + fmtNum(hi) + ']';
    }

    // ═══════════════════════════════════════════════════════════
    //  Data loading
    // ═══════════════════════════════════════════════════════════

    async function loadCompare() {
        try {
            if (runIds.length < 2) {
                showError('At least 2 run IDs are required for comparison. ' +
                    'Use <code>?run_ids=id1,id2[,id3...]</code> in the URL.');
                return;
            }

            // Fetch compare data
            var compareRes = await fetch('/api/compare?run_ids=' + encodeURIComponent(runIdsRaw));
            if (!compareRes.ok) {
                var errBody;
                try { errBody = await compareRes.json(); } catch (_) { errBody = null; }
                var errMsg = errBody && errBody.detail ? errBody.detail : ('HTTP ' + compareRes.status);
                showError(R.esc(errMsg));
                return;
            }
            compareData = await compareRes.json();

            // Fetch per-run stats in parallel (for error bars on grouped bar chart)
            var statPromises = compareData.runs.map(function (run) {
                return fetch('/api/run/' + encodeURIComponent(run.run_id) + '/stats')
                    .then(function (res) {
                        if (res.ok) return res.json().then(function (s) { return { id: run.run_id, stats: s }; });
                        return { id: run.run_id, stats: null };
                    })
                    .catch(function () { return { id: run.run_id, stats: null }; });
            });
            var statResults = await Promise.all(statPromises);
            statResults.forEach(function (sr) {
                perRunStats[sr.id] = sr.stats;
            });

            // Show content
            loadingEl.style.display = 'none';
            contentEl.style.display = '';

            var serverNames = compareData.runs.map(function (r) { return r.server_name || r.run_id.slice(0, 8); });
            compareTitleEl.textContent = 'Comparing ' + serverNames.join(', ');
            document.title = 'Compare ' + serverNames.join(', ') + ' — rag-bench';
            compareMetaEl.textContent = compareData.runs.length + ' runs selected — ' + runIdsRaw;

            // Render all charts
            renderRadar();
            renderGroupedBar();
            renderHeatmap();
            renderPairwiseTable();

        } catch (e) {
            showError('Could not load comparison data. Please check your connection and try again.');
            console.error(e);
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  (a) Multi-server radar chart
    // ═══════════════════════════════════════════════════════════

    function renderRadar() {
        var runs = compareData.runs;
        var N = runs.length;

        // Radar axes: Hit@5, MRR, and normalized Speed
        // Speed: 1 - latency/max_latency (higher is better on radar)
        var maxLatency = 1;
        runs.forEach(function (r) {
            var v = safeNum(r.query_latency_p50_ms);
            if (v > maxLatency) maxLatency = v;
        });

        var radarLabels = ['Hit@5', 'MRR', 'Speed'];

        var traces = runs.map(function (run, i) {
            var hit5 = safeNum(run.hit_at_5);
            var mrr = safeNum(run.mrr);
            var speed = maxLatency > 0 ? Math.max(0, 1 - safeNum(run.query_latency_p50_ms) / maxLatency) : 0;
            var vals = [hit5, mrr, speed];

            // Close the polygon
            return {
                type: 'scatterpolar',
                r: vals.concat([vals[0]]),
                theta: radarLabels.concat([radarLabels[0]]),
                fill: 'toself',
                name: run.server_name || run.run_id.slice(0, 8),
                marker: { color: palette[i % palette.length] },
                line: { color: palette[i % palette.length] },
                opacity: N > 4 ? 0.6 : 0.75
            };
        });

        var layout = plotlyLayout({
            title: { text: 'Multi-Server Radar (' + N + ' runs)', font: { size: 15 } },
            polar: {
                bgcolor: '#161b22',
                radialaxis: {
                    range: [0, 1],
                    tickfont: { color: '#8b949e', size: 11 },
                    tickformat: '.0%',
                    gridcolor: '#30363d',
                    showline: false
                },
                angularaxis: {
                    tickfont: { color: '#c9d1d9', size: 12 },
                    gridcolor: '#30363d'
                }
            },
            showlegend: true,
            legend: {
                x: 1.05,
                y: 0.5,
                font: { color: '#c9d1d9', size: 11 }
            },
            margin: { l: 30, r: 30, t: 60, b: 30 }
        });

        Plotly.newPlot('radarChart', traces, layout, PLOTLY_CONFIG);
    }

    // ═══════════════════════════════════════════════════════════
    //  (b) Grouped bar chart with error bars
    // ═══════════════════════════════════════════════════════════

    function renderGroupedBar() {
        var runs = compareData.runs;
        var N = runs.length;

        // One group per metric, one bar per run within group
        var traces = [];
        var metricsForBar = COMPARE_METRICS;

        metricsForBar.forEach(function (metric, mi) {
            var xVals = [];
            var yVals = [];
            var errHi = [];
            var errLo = [];

            runs.forEach(function (run, ri) {
                var val = safeNum(run[metric], 0);
                xVals.push(run.server_name || run.run_id.slice(0, 8));
                yVals.push(val);

                // Try to get CI from per-run stats
                var stats = perRunStats[run.run_id];
                var ciLow = null;
                var ciHigh = null;

                if (stats && stats.metrics) {
                    // Map compare metric names to stats metric names
                    var statsMetric = metric;
                    if (metric === 'query_latency_p50_ms') {
                        // Per-run stats use 'latency_ms' for the mean latency
                        statsMetric = 'latency_ms';
                    }
                    var m = stats.metrics[statsMetric];
                    if (m) {
                        ciLow = safeNum(m.ci_low, null);
                        ciHigh = safeNum(m.ci_high, null);
                    }
                }

                if (ciLow !== null && ciHigh !== null && ciLow !== ciHigh) {
                    errHi.push(ciHigh - val);
                    errLo.push(val - ciLow);
                } else {
                    // No CI available — show zero error bars
                    errHi.push(0);
                    errLo.push(0);
                }
            });

            traces.push({
                type: 'bar',
                x: [METRIC_LABELS[metric] || metric],
                y: yVals,
                name: METRIC_LABELS[metric] || metric,
                marker: { color: palette[mi % palette.length] },
                error_y: {
                    type: 'data',
                    array: errHi,
                    arrayminus: errLo,
                    visible: true,
                    color: '#8b949e',
                    thickness: 1.5
                },
                text: yVals.map(function (v) {
                    return metric === 'query_latency_p50_ms' ? v.toFixed(0) + ' ms' : (v * 100).toFixed(1) + '%';
                }),
                textposition: 'outside',
                textfont: { color: '#c9d1d9', size: 10 }
            });
        });

        // For a grouped bar, we need x values that are the run names,
        // and each trace represents a metric. Let me restructure.

        // Actually, for grouped bar with barmode='group':
        // x = [metric1, metric2, metric3], and each trace is one run
        // This gives one group per metric, with bars for each run side by side.

        // Let me redo this approach:
        var barTraces = [];
        runs.forEach(function (run, ri) {
            var xVals = [];
            var yVals = [];
            var errHi = [];
            var errLo = [];
            var texts = [];

            COMPARE_METRICS.forEach(function (metric) {
                var val = safeNum(run[metric], 0);
                xVals.push(METRIC_LABELS[metric] || metric);
                yVals.push(val);

                // Try to get CI from per-run stats
                var stats = perRunStats[run.run_id];
                var ciLow = null;
                var ciHigh = null;

                if (stats && stats.metrics) {
                    var statsMetric = metric === 'query_latency_p50_ms' ? 'latency_ms' : metric;
                    var m = stats.metrics[statsMetric];
                    if (m) {
                        ciLow = safeNum(m.ci_low, null);
                        ciHigh = safeNum(m.ci_high, null);
                    }
                }

                if (ciLow !== null && ciHigh !== null && ciLow !== ciHigh) {
                    errHi.push(Math.max(0, ciHigh - val));
                    errLo.push(Math.max(0, val - ciLow));
                } else {
                    errHi.push(0);
                    errLo.push(0);
                }

                texts.push(metric === 'query_latency_p50_ms' ? val.toFixed(0) + ' ms' : (val * 100).toFixed(1) + '%');
            });

            barTraces.push({
                type: 'bar',
                x: xVals,
                y: yVals,
                name: run.server_name || run.run_id.slice(0, 8),
                marker: { color: palette[ri % palette.length] },
                error_y: {
                    type: 'data',
                    array: errHi,
                    arrayminus: errLo,
                    visible: true,
                    color: '#8b949e',
                    thickness: 1.5
                },
                text: texts,
                textposition: 'outside',
                textfont: { color: '#c9d1d9', size: 10 }
            });
        });

        var barLayout = plotlyLayout({
            barmode: 'group',
            title: { text: 'Metrics by Run with 95% CI', font: { size: 15 } },
            xaxis: {
                title: null,
                tickfont: { color: '#c9d1d9', size: 12 },
                gridcolor: '#30363d'
            },
            yaxis: {
                title: { text: 'Value', font: { size: 12 } },
                gridcolor: '#30363d'
            },
            legend: {
                orientation: 'h',
                y: 1.15,
                font: { color: '#c9d1d9', size: 11 }
            },
            margin: { l: 70, r: 30, t: 50, b: 80 }
        });

        Plotly.newPlot('groupedBarChart', barTraces, barLayout, PLOTLY_CONFIG);
    }

    // ═══════════════════════════════════════════════════════════
    //  (c) Heatmap matrix with best-per-metric highlight
    // ═══════════════════════════════════════════════════════════

    function renderHeatmap() {
        var runs = compareData.runs;
        var N = runs.length;
        var M = COMPARE_METRICS.length;

        // Build z matrix: N rows (runs) × M columns (metrics)
        // For query_latency_p50_ms: lower is better, but for heatmap consistency
        // we show the raw values and highlight the best (lowest) cell
        var z = [];
        var yLabels = [];
        var colorsPerMetric = [];

        // Compute best per metric and build z
        var bestIdx = {}; // metric → best run index

        COMPARE_METRICS.forEach(function (metric, mi) {
            var bestVal = metric === 'query_latency_p50_ms' ? Infinity : -Infinity;
            var bestRi = -1;

            runs.forEach(function (run, ri) {
                var val = safeNum(run[metric], 0);
                if (metric === 'query_latency_p50_ms') {
                    if (val < bestVal) { bestVal = val; bestRi = ri; }
                } else {
                    if (val > bestVal) { bestVal = val; bestRi = ri; }
                }
            });

            bestIdx[metric] = bestRi;
        });

        // Build z matrix
        runs.forEach(function (run, ri) {
            var row = [];
            COMPARE_METRICS.forEach(function (metric) {
                row.push(safeNum(run[metric], 0));
            });
            z.push(row);
            yLabels.push(run.server_name || run.run_id.slice(0, 8));
        });

        var xLabels = COMPARE_METRICS.map(function (m) { return METRIC_LABELS[m] || m; });

        // Build custom hover text
        var hoverText = z.map(function (row, ri) {
            return row.map(function (val, ci) {
                var metric = COMPARE_METRICS[ci];
                if (metric === 'query_latency_p50_ms') {
                    return yLabels[ri] + '<br>' + xLabels[ci] + ': ' + val.toFixed(0) + ' ms';
                }
                return yLabels[ri] + '<br>' + xLabels[ci] + ': ' + (val * 100).toFixed(1) + '%';
            });
        });

        // Build annotations for best-per-metric cells (gold star ★)
        var annotations = [];
        COMPARE_METRICS.forEach(function (metric, ci) {
            var ri = bestIdx[metric];
            if (ri >= 0) {
                annotations.push({
                    x: xLabels[ci],
                    y: yLabels[ri],
                    text: '★',
                    showarrow: false,
                    font: { color: '#ffd700', size: 18 },
                    xanchor: 'center',
                    yanchor: 'middle'
                });
            }
        });

        var heatTrace = {
            type: 'heatmap',
            y: yLabels,
            x: xLabels,
            z: z,
            text: hoverText,
            hovertemplate: '%{text}<extra></extra>',
            colorscale: [
                [0, '#161b22'],
                [0.5, '#1f6feb'],
                [1, '#58a6ff']
            ],
            colorbar: {
                title: { text: 'Value', font: { color: '#8b949e' } },
                tickfont: { color: '#8b949e' }
            },
            showscale: true
        };

        var heatLayout = plotlyLayout({
            title: { text: 'Metrics Matrix (' + N + ' servers × ' + M + ' metrics)', font: { size: 15 } },
            annotations: annotations,
            xaxis: {
                side: 'top',
                tickfont: { color: '#c9d1d9', size: 12 },
                gridcolor: '#30363d'
            },
            yaxis: {
                tickfont: { color: '#c9d1d9', size: 11 },
                gridcolor: '#30363d',
                automargin: true
            },
            margin: { l: 150, r: 30, t: 80, b: 60 }
        });

        Plotly.newPlot('heatmapChart', [heatTrace], heatLayout, PLOTLY_CONFIG);
    }

    // ═══════════════════════════════════════════════════════════
    //  (d) A/B pairwise table
    // ═══════════════════════════════════════════════════════════

    function renderPairwiseTable() {
        var runs = compareData.runs;
        var pairwise = compareData.pairwise_ab || {};
        var container = document.getElementById('pairwiseTable');
        container.innerHTML = '';

        if (!pairwise || Object.keys(pairwise).length === 0) {
            container.innerHTML = '<p style="color:#8b949e;padding:16px;">No pairwise comparison data available. Ensure all runs share overlapping query IDs.</p>';
            return;
        }

        // Build a run_id → server_name lookup
        var nameMap = {};
        runs.forEach(function (r) {
            nameMap[r.run_id] = r.server_name || r.run_id.slice(0, 8);
        });

        // Generate all C(N,2) pairs
        for (var i = 0; i < runs.length; i++) {
            for (var j = i + 1; j < runs.length; j++) {
                var runA = runs[i];
                var runB = runs[j];
                var pairKey = runA.run_id + ',' + runB.run_id;
                var pairData = pairwise[pairKey];

                var section = document.createElement('div');
                section.className = 'pair-section';

                var h3 = document.createElement('h3');
                h3.textContent = nameMap[runA.run_id] + '  vs  ' + nameMap[runB.run_id];
                section.appendChild(h3);

                var table = document.createElement('table');
                table.innerHTML = '<thead><tr>' +
                    '<th>Metric</th>' +
                    '<th>Delta</th>' +
                    '<th>95% CI</th>' +
                    '<th>p-value</th>' +
                    "<th>Cohen's d</th>" +
                    "<th>Cliff's delta</th>" +
                    '</tr></thead><tbody></tbody>';
                section.appendChild(table);

                var tbody = table.querySelector('tbody');

                COMPARE_METRICS.forEach(function (metric) {
                    var pd = pairData && pairData[metric] ? pairData[metric] : null;
                    var tr = document.createElement('tr');

                    var delta = pd ? pd.delta : null;
                    var ciLo = pd ? pd.delta_ci_lo : null;
                    var ciHi = pd ? pd.delta_ci_hi : null;
                    var pval = pd ? pd.p_value : null;
                    var cd = pd ? pd.cohens_d : null;
                    var cld = pd ? pd.cliffs_delta : null;

                    // Color the p-value cell
                    var pvalDisplay = fmtPval(pval);
                    var pvalStyle = '';
                    if (pval !== null && pval < 0.05) {
                        pvalStyle = 'color:#3fb950;font-weight:600;';
                    } else if (pval !== null && pval < 0.1) {
                        pvalStyle = 'color:#d29922;';
                    }

                    tr.innerHTML =
                        '<td style="font-weight:600;">' + R.esc(METRIC_LABELS[metric] || metric) + '</td>' +
                        '<td class="metric">' + fmtNum(delta) + '</td>' +
                        '<td class="metric">' + fmtCI(ciLo, ciHi) + '</td>' +
                        '<td class="metric" style="' + pvalStyle + '">' + pvalDisplay + '</td>' +
                        '<td class="metric">' + fmtNum(cd) + '</td>' +
                        '<td class="metric">' + fmtNum(cld) + '</td>';

                    tbody.appendChild(tr);
                });

                container.appendChild(section);
            }
        }
    }

    // ── Start loading ──
    if (runIds.length < 2) {
        showError('At least 2 run IDs are required for comparison. ' +
            'Use <code>?run_ids=id1,id2[,id3...]</code> in the URL.');
    } else {
        loadCompare();
    }
})();
