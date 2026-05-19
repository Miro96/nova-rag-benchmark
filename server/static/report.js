/* rag-bench single-run report — Plotly visualizations for /run/{id}/report */
(function () {
    'use strict';

    var R = window.RagBench;
    var palette = R.palette;
    var plotlyLayout = R.plotlyLayout;

    // ── Extract run_id from URL path ──
    var pathParts = window.location.pathname.split('/');
    var runId = pathParts[2] || '';

    // ── DOM refs ──
    var loadingEl = document.getElementById('loadingState');
    var errorEl = document.getElementById('errorState');
    var errorDetailEl = document.getElementById('errorDetail');
    var reportContentEl = document.getElementById('reportContent');
    var reportTitleEl = document.getElementById('reportTitle');
    var runMetaEl = document.getElementById('runMeta');
    var tabs = document.querySelectorAll('[role="tab"]');
    var panels = document.querySelectorAll('[role="tabpanel"]');

    // ── State ──
    var runData = null;
    var queriesData = null;
    var statsData = null;
    var baselineStatsData = null; // baseline run's stats (for radar)
    var baselineRunData = null;   // baseline run's run data
    var renderedTabs = {};        // track lazy-rendered tabs

    // ── Metrics for display ──
    var DISPLAY_METRICS = ['hit_at_5', 'chunk_hit_at_5', 'symbol_hit_at_5', 'mrr'];
    var DISPLAY_LABELS = {
        hit_at_5: 'Hit@5',
        chunk_hit_at_5: 'Chunk Hit@5',
        symbol_hit_at_5: 'Symbol Hit@5',
        mrr: 'MRR',
        latency_ms: 'Latency (ms)',
        response_tokens: 'Resp. Tokens'
    };

    // ── Tab switching ──
    function activateTab(tabId) {
        var targetPanelId = null;
        tabs.forEach(function (t) {
            if (t.id === tabId) {
                t.setAttribute('aria-selected', 'true');
                targetPanelId = t.getAttribute('aria-controls');
            } else {
                t.setAttribute('aria-selected', 'false');
            }
        });
        panels.forEach(function (p) {
            if (p.id === targetPanelId) {
                p.classList.add('visible');
                // Resize any Plotly plots that were rendered hidden
                var gd = p.querySelector('.js-plotly-plot');
                if (gd && window.Plotly) {
                    Plotly.Plots.resize(gd);
                }
            } else {
                p.classList.remove('visible');
            }
        });

        // Lazy-render the activated tab's charts
        if (targetPanelId && !renderedTabs[targetPanelId] && statsData) {
            renderedTabs[targetPanelId] = true;
            switch (targetPanelId) {
                case 'panel-overview':
                    renderOverview();
                    break;
                case 'panel-breakdowns':
                    renderBreakdowns();
                    break;
                case 'panel-latency':
                    renderLatency();
                    break;
                case 'panel-heatmap':
                    renderHeatmap();
                    break;
                case 'panel-statistics':
                    renderStatistics();
                    break;
            }
        }
    }

    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            activateTab(tab.id);
        });
        tab.addEventListener('keydown', function (e) {
            var idx = Array.prototype.indexOf.call(tabs, tab);
            var target = null;
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                target = tabs[(idx + 1) % tabs.length];
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                target = tabs[(idx - 1 + tabs.length) % tabs.length];
            } else if (e.key === 'Home') {
                e.preventDefault();
                target = tabs[0];
            } else if (e.key === 'End') {
                e.preventDefault();
                target = tabs[tabs.length - 1];
            }
            if (target) {
                target.focus();
                activateTab(target.id);
            }
        });
    });

    // ── Error state ──
    function showError(msg) {
        loadingEl.style.display = 'none';
        errorEl.style.display = 'block';
        reportContentEl.style.display = 'none';
        errorDetailEl.innerHTML = msg;
    }

    // ── Helper: safely get a prop, default 0 ──
    function safeNum(v, d) {
        d = d === undefined ? 0 : d;
        if (v === null || v === undefined || isNaN(v)) return d;
        return v;
    }

    // ── Helper: build chart container div ──
    function chartDiv(id) {
        var d = document.createElement('div');
        d.id = id;
        d.style.width = '100%';
        return d;
    }

    // ── Plotly config (common) ──
    var PLOTLY_CONFIG = {
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        responsive: true
    };

    // ═══════════════════════════════════════════════════════════
    //  Data loading
    // ═══════════════════════════════════════════════════════════

    async function loadRun() {
        try {
            // Fetch run data
            var runRes = await fetch('/api/run/' + encodeURIComponent(runId));
            if (!runRes.ok) {
                showError('Run &ldquo;' + R.esc(runId) + '&rdquo; was not found on this leaderboard. It may have been removed or the URL may be mistyped.');
                return;
            }
            runData = await runRes.json();

            // Fetch queries and stats in parallel
            var qRes = fetch('/api/run/' + encodeURIComponent(runId) + '/queries');
            var sRes = fetch('/api/run/' + encodeURIComponent(runId) + '/stats');

            var qResp = await qRes;
            if (qResp.ok) {
                queriesData = await qResp.json();
            } else {
                queriesData = [];
            }

            var sResp = await sRes;
            if (sResp.ok) {
                statsData = await sResp.json();
            } else {
                statsData = null;
            }

            // Fetch baseline run's data & stats if baseline_ab is present
            if (statsData && statsData.baseline_ab && statsData.baseline_ab.baseline_run_id) {
                var baselineId = statsData.baseline_ab.baseline_run_id;
                try {
                    var blRunRes = await fetch('/api/run/' + encodeURIComponent(baselineId));
                    if (blRunRes.ok) baselineRunData = await blRunRes.json();
                    var blStatsRes = await fetch('/api/run/' + encodeURIComponent(baselineId) + '/stats');
                    if (blStatsRes.ok) baselineStatsData = await blStatsRes.json();
                } catch (_) {
                    // baseline fetch failed — radar will show single trace
                }
            }

            // Show report
            loadingEl.style.display = 'none';
            reportContentEl.style.display = '';

            reportTitleEl.textContent = runData.server_name || 'Run Report';
            var submitted = runData.submitted_at || '';
            document.title = (runData.server_name || 'Run') + ' Report — rag-bench';
            runMetaEl.textContent = 'Run ' + R.esc(runId) + (submitted ? ' — submitted ' + R.esc(submitted) : '');

            // Render Overview immediately (it's the visible tab)
            renderedTabs['panel-overview'] = true;
            renderOverview();

        } catch (e) {
            showError('Could not load run data. Please check your connection and try again.');
            console.error(e);
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Overview Tab: bar chart + gauge + radar
    // ═══════════════════════════════════════════════════════════

    function renderOverview() {
        var panel = document.getElementById('panel-overview');
        panel.innerHTML = '';

        // --- Export buttons row ---
        var exportRow = document.createElement('div');
        exportRow.style.cssText = 'display:flex;gap:8px;margin-bottom:12px;justify-content:flex-end;';
        var exportPngBtn = document.createElement('button');
        exportPngBtn.className = 'btn';
        exportPngBtn.textContent = 'Export PNG';
        exportPngBtn.addEventListener('click', function () {
            exportChart('overview-bar', 'png', runData.server_name + '-overview-metrics');
        });
        var exportSvgBtn = document.createElement('button');
        exportSvgBtn.className = 'btn';
        exportSvgBtn.textContent = 'Export SVG';
        exportSvgBtn.addEventListener('click', function () {
            exportChart('overview-bar', 'svg', runData.server_name + '-overview-metrics');
        });
        exportRow.appendChild(exportPngBtn);
        exportRow.appendChild(exportSvgBtn);
        panel.appendChild(exportRow);

        // --- Metrics bar chart ---
        var barDiv = chartDiv('overview-bar');
        panel.appendChild(barDiv);
        renderMetricsBar(barDiv);

        // --- Composite score gauge ---
        var gaugeDiv = chartDiv('overview-gauge');
        gaugeDiv.style.maxWidth = '400px';
        gaugeDiv.style.margin = '24px auto 0 auto';
        panel.appendChild(gaugeDiv);
        renderCompositeGauge(gaugeDiv);

        // --- Radar chart ---
        var radarDiv = chartDiv('overview-radar');
        radarDiv.style.marginTop = '24px';
        panel.appendChild(radarDiv);
        renderRadar(radarDiv);
    }

    // ── Chart export helper ──
    function exportChart(divId, format, baseName) {
        var gd = document.getElementById(divId);
        if (!gd) return;
        Plotly.toImage(gd, { format: format, width: 1200, height: 800 }).then(function (dataUrl) {
            var a = document.createElement('a');
            a.download = baseName + '.' + format;
            a.href = dataUrl;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }).catch(function (err) {
            console.error('Export failed:', err);
        });
    }

    function renderMetricsBar(div) {
        if (!statsData || !statsData.metrics) return;

        var metrics = DISPLAY_METRICS;
        var x = [];
        var y = [];
        var errHi = [];
        var errLo = [];
        var colors = [];

        metrics.forEach(function (m, i) {
            var d = statsData.metrics[m];
            if (!d) return;
            x.push(DISPLAY_LABELS[m] || m);
            y.push(safeNum(d.point));
            errHi.push(safeNum(d.ci_high) - safeNum(d.point));
            errLo.push(safeNum(d.point) - safeNum(d.ci_low));
            colors.push(palette[i % palette.length]);
        });

        var trace = {
            type: 'bar',
            x: x,
            y: y,
            marker: { color: colors },
            error_y: {
                type: 'data',
                array: errHi,
                arrayminus: errLo,
                visible: true,
                color: '#c9d1d9'
            },
            text: y.map(function (v) { return R.pct(v); }),
            textposition: 'outside',
            textfont: { color: '#c9d1d9' }
        };

        var layout = plotlyLayout({
            title: { text: 'Top Metrics with 95% CI', font: { size: 15 } },
            xaxis: { title: null, gridcolor: '#30363d' },
            yaxis: {
                title: { text: 'Score', font: { size: 12 } },
                range: [0, Math.max(1.0, Math.max.apply(null, y) * 1.2)],
                tickformat: '.0%',
                gridcolor: '#30363d'
            },
            margin: { l: 70, r: 30, t: 50, b: 80 }
        });

        Plotly.newPlot(div, [trace], layout, PLOTLY_CONFIG);
    }

    function renderCompositeGauge(div) {
        var score = safeNum(runData.composite_score, 0);
        var color = score >= 70 ? '#3fb950' : score >= 40 ? '#d29922' : '#f85149';

        var trace = {
            type: 'indicator',
            mode: 'gauge+number',
            value: score,
            number: {
                font: { color: color, size: 48 },
                suffix: '%'
            },
            gauge: {
                axis: {
                    range: [0, 100],
                    tickfont: { color: '#8b949e', size: 11 },
                    tickcolor: '#30363d'
                },
                bar: { color: color, thickness: 0.2 },
                bgcolor: '#161b22',
                borderwidth: 0,
                steps: [
                    { range: [0, 40], color: 'rgba(248,81,73,0.15)' },
                    { range: [40, 70], color: 'rgba(210,153,34,0.15)' },
                    { range: [70, 100], color: 'rgba(63,185,80,0.15)' }
                ],
                threshold: {
                    line: { color: '#c9d1d9', width: 1 },
                    thickness: 0.9,
                    value: score
                }
            }
        };

        var layout = plotlyLayout({
            title: { text: 'Composite Score', font: { size: 15 } },
            margin: { l: 20, r: 20, t: 50, b: 20 }
        });

        Plotly.newPlot(div, [trace], layout, PLOTLY_CONFIG);
    }

    function renderRadar(div) {
        if (!statsData || !statsData.metrics) return;

        // Radar axes (all proportion metrics)
        var radarMetrics = ['hit_at_5', 'chunk_hit_at_5', 'symbol_hit_at_5', 'mrr'];
        var radarLabels = radarMetrics.map(function (m) { return DISPLAY_LABELS[m] || m; });

        // Current run's values
        var runVals = radarMetrics.map(function (m) {
            var d = statsData.metrics[m];
            return d ? safeNum(d.point) : 0;
        });

        var traces = [{
            type: 'scatterpolar',
            r: runVals.concat([runVals[0]]), // close the polygon
            theta: radarLabels.concat([radarLabels[0]]),
            fill: 'toself',
            name: runData.server_name || 'Current Run',
            marker: { color: palette[0] },
            line: { color: palette[0] }
        }];

        // Baseline trace (if available)
        if (baselineStatsData && baselineStatsData.metrics && baselineRunData) {
            var blVals = radarMetrics.map(function (m) {
                var d = baselineStatsData.metrics[m];
                return d ? safeNum(d.point) : 0;
            });
            traces.push({
                type: 'scatterpolar',
                r: blVals.concat([blVals[0]]),
                theta: radarLabels.concat([radarLabels[0]]),
                fill: 'toself',
                name: baselineRunData.server_name || 'Baseline',
                marker: { color: palette[1] },
                line: { color: palette[1] },
                opacity: 0.7
            });
        }

        var layout = plotlyLayout({
            title: { text: 'Run vs Baseline Radar', font: { size: 15 } },
            polar: {
                bgcolor: '#161b22',
                radialaxis: {
                    range: [0, 1],
                    tickfont: { color: '#8b949e', size: 11 },
                    tickformat: '.0%',
                    gridcolor: '#30363d'
                },
                angularaxis: {
                    tickfont: { color: '#c9d1d9', size: 12 },
                    gridcolor: '#30363d'
                }
            },
            showlegend: true,
            legend: { x: 1.05, y: 0.5 },
            margin: { l: 30, r: 30, t: 50, b: 30 }
        });

        Plotly.newPlot(div, traces, layout, PLOTLY_CONFIG);
    }

    // ═══════════════════════════════════════════════════════════
    //  Breakdowns Tab: difficulty / type / repo bar charts
    // ═══════════════════════════════════════════════════════════

    function renderBreakdowns() {
        var panel = document.getElementById('panel-breakdowns');
        panel.innerHTML = '';

        var sections = [
            { key: 'by_difficulty', title: 'By Difficulty', bucketLabel: 'Difficulty' },
            { key: 'by_type', title: 'By Type', bucketLabel: 'Type' },
            { key: 'by_repo', title: 'By Repo', bucketLabel: 'Repo' }
        ];

        sections.forEach(function (sec) {
            var h3 = document.createElement('h3');
            h3.textContent = sec.title;
            h3.style.cssText = 'color:#8b949e;font-size:14px;margin:20px 0 8px 0;';
            panel.appendChild(h3);

            var div = chartDiv('breakdown-' + sec.key);
            panel.appendChild(div);

            renderBucketBar(div, sec.key, sec.title);
        });
    }

    function renderBucketBar(div, bucketKey, title) {
        if (!statsData || !statsData[bucketKey]) {
            div.innerHTML = '<p style="color:#8b949e;padding:16px;">No data available.</p>';
            return;
        }

        var buckets = statsData[bucketKey];
        var bucketNames = Object.keys(buckets);
        if (bucketNames.length === 0) {
            div.innerHTML = '<p style="color:#8b949e;padding:16px;">No buckets found.</p>';
            return;
        }

        // Build grouped bar: one group per metric, one bar per bucket
        var traces = [];
        DISPLAY_METRICS.forEach(function (metric, mi) {
            var xVals = [];
            var yVals = [];
            var errHi = [];
            var errLo = [];

            bucketNames.forEach(function (bn) {
                var d = buckets[bn] && buckets[bn][metric];
                if (d && d.n > 0) {
                    xVals.push(bn);
                    yVals.push(safeNum(d.point));
                    errHi.push(safeNum(d.ci_high) - safeNum(d.point));
                    errLo.push(safeNum(d.point) - safeNum(d.ci_low));
                }
            });

            if (xVals.length > 0) {
                traces.push({
                    type: 'bar',
                    x: xVals,
                    y: yVals,
                    name: DISPLAY_LABELS[metric] || metric,
                    marker: { color: palette[mi % palette.length] },
                    error_y: {
                        type: 'data',
                        array: errHi,
                        arrayminus: errLo,
                        visible: true,
                        color: '#8b949e'
                    }
                });
            }
        });

        var layout = plotlyLayout({
            barmode: 'group',
            xaxis: {
                title: null,
                tickfont: { color: '#c9d1d9', size: 11 },
                gridcolor: '#30363d'
            },
            yaxis: {
                title: { text: 'Score', font: { size: 12 } },
                tickformat: '.0%',
                gridcolor: '#30363d'
            },
            legend: { orientation: 'h', y: 1.12 },
            margin: { l: 70, r: 30, t: 30, b: 100 }
        });

        Plotly.newPlot(div, traces, layout, PLOTLY_CONFIG);
    }

    // ═══════════════════════════════════════════════════════════
    //  Latency Tab: histogram + box plot
    // ═══════════════════════════════════════════════════════════

    function renderLatency() {
        var panel = document.getElementById('panel-latency');
        panel.innerHTML = '';

        if (!queriesData || queriesData.length === 0) {
            panel.innerHTML = '<p style="color:#8b949e;padding:16px;">No query data available.</p>';
            return;
        }

        var latencies = queriesData.map(function (q) { return safeNum(q.latency_ms); });

        // --- Histogram ---
        var histH3 = document.createElement('h3');
        histH3.textContent = 'Latency Distribution';
        histH3.style.cssText = 'color:#8b949e;font-size:14px;margin:0 0 8px 0;';
        panel.appendChild(histH3);
        var histDiv = chartDiv('latency-histogram');
        panel.appendChild(histDiv);

        var histTrace = {
            type: 'histogram',
            x: latencies,
            marker: { color: palette[0], line: { color: '#30363d', width: 1 } },
            nbinsx: Math.min(20, Math.ceil(Math.sqrt(latencies.length)))
        };

        Plotly.newPlot(histDiv, [histTrace], plotlyLayout({
            title: { text: 'Latency (ms) Histogram', font: { size: 14 } },
            xaxis: { title: { text: 'Latency (ms)' }, gridcolor: '#30363d' },
            yaxis: { title: { text: 'Count' }, gridcolor: '#30363d' },
            margin: { l: 60, r: 30, t: 50, b: 60 }
        }), PLOTLY_CONFIG);

        // --- Box plot by query_type ---
        var boxH3 = document.createElement('h3');
        boxH3.textContent = 'Latency by Query Type';
        boxH3.style.cssText = 'color:#8b949e;font-size:14px;margin:24px 0 8px 0;';
        panel.appendChild(boxH3);
        var boxDiv = chartDiv('latency-box');
        panel.appendChild(boxDiv);

        // Group latencies by query_type
        var typeGroups = {};
        queriesData.forEach(function (q) {
            var t = q.type || 'unknown';
            if (!typeGroups[t]) typeGroups[t] = [];
            typeGroups[t].push(safeNum(q.latency_ms));
        });

        var boxTrace = {
            type: 'box',
            y: Object.values(typeGroups).flat(),
            x: Object.keys(typeGroups).reduce(function (acc, t) {
                var arr = typeGroups[t];
                for (var i = 0; i < arr.length; i++) acc.push(t);
                return acc;
            }, []),
            marker: { color: palette[1] },
            line: { color: '#c9d1d9' },
            boxpoints: 'outliers',
            jitter: 0.5
        };

        Plotly.newPlot(boxDiv, [boxTrace], plotlyLayout({
            title: { text: 'Latency by Query Type', font: { size: 14 } },
            xaxis: { title: { text: 'Query Type' }, gridcolor: '#30363d' },
            yaxis: { title: { text: 'Latency (ms)' }, gridcolor: '#30363d' },
            margin: { l: 70, r: 30, t: 50, b: 80 }
        }), PLOTLY_CONFIG);
    }

    // ═══════════════════════════════════════════════════════════
    //  Per-Query Heatmap Tab
    // ═══════════════════════════════════════════════════════════

    // ── Heatmap state (persisted across filter updates for Plotly.react) ──
    var heatmapInitialized = false;
    var heatmapDiffSelect = null;
    var heatmapTypeSelect = null;
    var heatmapNoDataP = null;

    function renderHeatmap(difficultyFilter, typeFilter) {
        var panel = document.getElementById('panel-heatmap');

        // Normalize filters
        difficultyFilter = difficultyFilter || 'all';
        typeFilter = typeFilter || 'all';

        if (heatmapInitialized) {
            // ── Filter-change path (Plotly.react) ──
            var filtered = queriesData.filter(function (q) {
                if (difficultyFilter !== 'all' && (q.difficulty || 'unknown') !== difficultyFilter) return false;
                if (typeFilter !== 'all' && (q.type || 'unknown') !== typeFilter) return false;
                return true;
            });

            if (filtered.length === 0) {
                document.getElementById('heatmap-plot').style.display = 'none';
                if (heatmapNoDataP) heatmapNoDataP.style.display = '';
                if (heatmapNoDataP) heatmapNoDataP.textContent = 'No queries match the selected filters.';
                return;
            }

            document.getElementById('heatmap-plot').style.display = '';
            if (heatmapNoDataP) heatmapNoDataP.style.display = 'none';

            var maxLatency = 1;
            var maxTokens = 1;
            filtered.forEach(function (q) {
                var l = safeNum(q.latency_ms);
                var t = safeNum(q.response_tokens);
                if (l > maxLatency) maxLatency = l;
                if (t > maxTokens) maxTokens = t;
            });

            var z = filtered.map(function (q) {
                return [
                    q.found_file ? 1 : 0,
                    q.found_symbol ? 1 : 0,
                    q.found_chunk ? 1 : 0,
                    safeNum(q.latency_ms) / (maxLatency || 1),
                    safeNum(q.response_tokens) / (maxTokens || 1)
                ];
            });

            var data = [{
                type: 'heatmap',
                y: filtered.map(function (q) { return q.id || '?'; }),
                x: ['Found File', 'Found Symbol', 'Found Chunk', 'Latency (norm)', 'Tokens (norm)'],
                z: z,
                colorscale: [
                    [0, '#161b22'],
                    [0.5, '#1f6feb'],
                    [1, '#3fb950']
                ],
                hovertemplate: 'Query: %{y}<br>%{x}: %{z:.2f}<extra></extra>',
                colorbar: {
                    title: { text: 'Value', font: { color: '#8b949e' } },
                    tickfont: { color: '#8b949e' }
                }
            }];

            var layout = plotlyLayout({
                title: { text: 'Per-Query Metrics (n=' + filtered.length + ')', font: { size: 14 } },
                xaxis: { side: 'top', tickfont: { color: '#c9d1d9', size: 11 }, gridcolor: '#30363d' },
                yaxis: { tickfont: { color: '#8b949e', size: 10 }, gridcolor: '#30363d', automargin: true },
                margin: { l: 120, r: 30, t: 80, b: 60 }
            });

            Plotly.react('heatmap-plot', data, layout, PLOTLY_CONFIG);
            return;
        }

        // ── Initial setup ──
        panel.innerHTML = '';

        if (!queriesData || queriesData.length === 0) {
            panel.innerHTML = '<p style="color:#8b949e;padding:16px;">No query data available.</p>';
            return;
        }

        // Build filter controls
        var controlsDiv = document.createElement('div');
        controlsDiv.className = 'controls';
        controlsDiv.style.cssText = 'display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;align-items:center;';

        // Difficulty filter
        var diffValues = ['all'];
        queriesData.forEach(function (q) {
            var d = q.difficulty || 'unknown';
            if (diffValues.indexOf(d) === -1) diffValues.push(d);
        });
        var diffLabel = document.createElement('label');
        diffLabel.style.cssText = 'color:#8b949e;font-size:13px;';
        diffLabel.textContent = 'Difficulty: ';
        heatmapDiffSelect = document.createElement('select');
        diffValues.forEach(function (v) {
            var opt = document.createElement('option');
            opt.value = v;
            opt.textContent = v.charAt(0).toUpperCase() + v.slice(1);
            if (v === difficultyFilter) opt.selected = true;
            heatmapDiffSelect.appendChild(opt);
        });
        diffLabel.appendChild(heatmapDiffSelect);
        controlsDiv.appendChild(diffLabel);

        // Type filter
        var typeValues = ['all'];
        queriesData.forEach(function (q) {
            var t = q.type || 'unknown';
            if (typeValues.indexOf(t) === -1) typeValues.push(t);
        });
        var typeLabel = document.createElement('label');
        typeLabel.style.cssText = 'color:#8b949e;font-size:13px;';
        typeLabel.textContent = 'Type: ';
        heatmapTypeSelect = document.createElement('select');
        typeValues.forEach(function (v) {
            var opt = document.createElement('option');
            opt.value = v;
            opt.textContent = v.charAt(0).toUpperCase() + v.slice(1);
            if (v === typeFilter) opt.selected = true;
            heatmapTypeSelect.appendChild(opt);
        });
        typeLabel.appendChild(heatmapTypeSelect);
        controlsDiv.appendChild(typeLabel);

        panel.appendChild(controlsDiv);

        // No-data message (hidden by default)
        heatmapNoDataP = document.createElement('p');
        heatmapNoDataP.style.cssText = 'color:#8b949e;padding:16px;display:none;';
        heatmapNoDataP.textContent = 'No queries match the selected filters.';
        panel.appendChild(heatmapNoDataP);

        // Filter queries
        var filtered = queriesData.filter(function (q) {
            if (difficultyFilter !== 'all' && (q.difficulty || 'unknown') !== difficultyFilter) return false;
            if (typeFilter !== 'all' && (q.type || 'unknown') !== typeFilter) return false;
            return true;
        });

        if (filtered.length === 0) {
            heatmapNoDataP.style.display = '';
        }

        // Heatmap: y = query_ids, x = metric columns
        var yLabels = filtered.map(function (q) { return q.id || '?'; });
        var xLabels = ['found_file', 'found_symbol', 'found_chunk', 'latency_n', 'tokens_n'];

        // Compute z values (normalized latency and tokens)
        var maxLatency = 1;
        var maxTokens = 1;
        filtered.forEach(function (q) {
            var l = safeNum(q.latency_ms);
            var t = safeNum(q.response_tokens);
            if (l > maxLatency) maxLatency = l;
            if (t > maxTokens) maxTokens = t;
        });

        var z = [];
        filtered.forEach(function (q) {
            z.push([
                q.found_file ? 1 : 0,
                q.found_symbol ? 1 : 0,
                q.found_chunk ? 1 : 0,
                safeNum(q.latency_ms) / (maxLatency || 1),
                safeNum(q.response_tokens) / (maxTokens || 1)
            ]);
        });

        var heatmapDiv = chartDiv('heatmap-plot');
        panel.appendChild(heatmapDiv);

        var heatTrace = {
            type: 'heatmap',
            y: yLabels,
            x: ['Found File', 'Found Symbol', 'Found Chunk', 'Latency (norm)', 'Tokens (norm)'],
            z: z,
            colorscale: [
                [0, '#161b22'],
                [0.5, '#1f6feb'],
                [1, '#3fb950']
            ],
            hovertemplate: 'Query: %{y}<br>%{x}: %{z:.2f}<extra></extra>',
            colorbar: {
                title: { text: 'Value', font: { color: '#8b949e' } },
                tickfont: { color: '#8b949e' }
            }
        };

        Plotly.newPlot(heatmapDiv, [heatTrace], plotlyLayout({
            title: { text: 'Per-Query Metrics (n=' + filtered.length + ')', font: { size: 14 } },
            xaxis: { side: 'top', tickfont: { color: '#c9d1d9', size: 11 }, gridcolor: '#30363d' },
            yaxis: { tickfont: { color: '#8b949e', size: 10 }, gridcolor: '#30363d', automargin: true },
            margin: { l: 120, r: 30, t: 80, b: 60 }
        }), PLOTLY_CONFIG);

        // Wire filter change events
        heatmapDiffSelect.addEventListener('change', function () {
            renderHeatmap(heatmapDiffSelect.value, heatmapTypeSelect.value);
        });
        heatmapTypeSelect.addEventListener('change', function () {
            renderHeatmap(heatmapDiffSelect.value, heatmapTypeSelect.value);
        });

        heatmapInitialized = true;
    }

    // ═══════════════════════════════════════════════════════════
    //  Statistics Tab: CI/CV/IQR table + correlation heatmap + A/B table
    // ═══════════════════════════════════════════════════════════

    function renderStatistics() {
        var panel = document.getElementById('panel-statistics');
        panel.innerHTML = '';

        if (!statsData) {
            panel.innerHTML = '<p style="color:#8b949e;padding:16px;">No statistics data available.</p>';
            return;
        }

        // ── 1. CI / CV / IQR table ──
        var section1 = document.createElement('div');
        section1.style.marginBottom = '32px';

        var h3a = document.createElement('h3');
        h3a.textContent = 'Metric Confidence Intervals, CV & IQR';
        h3a.style.cssText = 'color:#8b949e;font-size:14px;margin:0 0 12px 0;';
        section1.appendChild(h3a);

        var table1 = document.createElement('table');
        table1.innerHTML = '<thead><tr>' +
            '<th>Metric</th><th>Point</th><th>CI Low</th><th>CI High</th><th>CV</th><th>IQR</th>' +
            '</tr></thead><tbody></tbody>';
        section1.appendChild(table1);

        var tbody = table1.querySelector('tbody');
        var allMetrics = DISPLAY_METRICS.concat(['latency_ms', 'response_tokens']);
        allMetrics.forEach(function (m) {
            var d = statsData.metrics && statsData.metrics[m];
            if (!d) return;
            var cv = (statsData.cv && statsData.cv[m] != null) ? statsData.cv[m] : null;
            var iqr = (statsData.iqr && statsData.iqr[m] != null) ? statsData.iqr[m] : null;
            var tr = document.createElement('tr');
            tr.innerHTML = '<td style="font-weight:600;">' + R.esc(DISPLAY_LABELS[m] || m) + '</td>' +
                '<td class="metric">' + formatNum(d.point) + '</td>' +
                '<td class="metric">' + formatNum(d.ci_low) + '</td>' +
                '<td class="metric">' + formatNum(d.ci_high) + '</td>' +
                '<td class="metric">' + (cv != null ? formatNum(cv) : '—') + '</td>' +
                '<td class="metric">' + (iqr != null ? formatNum(iqr) : '—') + '</td>';
            tbody.appendChild(tr);
        });

        panel.appendChild(section1);

        // ── 2. Correlation heatmap ──
        var h3b = document.createElement('h3');
        h3b.textContent = 'Metric Correlation Matrix';
        h3b.style.cssText = 'color:#8b949e;font-size:14px;margin:0 0 8px 0;';
        panel.appendChild(h3b);

        if (statsData.correlations && statsData.correlations.matrix && statsData.correlations.matrix.length > 0) {
            var corrDiv = chartDiv('correlation-heatmap');
            panel.appendChild(corrDiv);

            var corrLabels = statsData.correlations.labels.map(function (l) {
                return DISPLAY_LABELS[l] || l;
            });
            // Sanitize: replace JSON null with undefined so Plotly ignores those cells
            var corrMatrix = statsData.correlations.matrix.map(function (row) {
                return row.map(function (v) { return v != null ? v : undefined; });
            });

            var corrTrace = {
                type: 'heatmap',
                y: corrLabels,
                x: corrLabels,
                z: corrMatrix,
                zmin: -1,
                zmax: 1,
                colorscale: [
                    [0, '#f85149'],
                    [0.5, '#161b22'],
                    [1, '#3fb950']
                ],
                text: corrMatrix.map(function (row) {
                    return row.map(function (v) { return v != null ? v.toFixed(3) : '---'; });
                }),
                texttemplate: '%{text}',
                textfont: { color: '#c9d1d9', size: 10 },
                hovertemplate: '%{x} vs %{y}: %{z:.4f}<extra></extra>',
                colorbar: {
                    title: { text: 'r', font: { color: '#8b949e' } },
                    tickfont: { color: '#8b949e' }
                }
            };

            Plotly.newPlot(corrDiv, [corrTrace], plotlyLayout({
                title: { text: 'Pearson Correlation', font: { size: 14 } },
                xaxis: { side: 'top', tickfont: { color: '#c9d1d9', size: 10 }, gridcolor: '#30363d' },
                yaxis: { tickfont: { color: '#c9d1d9', size: 10 }, gridcolor: '#30363d', automargin: true },
                margin: { l: 130, r: 30, t: 80, b: 60 }
            }), PLOTLY_CONFIG);
        } else {
            panel.appendChild(document.createElement('p'));
            panel.lastChild.style.cssText = 'color:#8b949e;padding:16px;';
            panel.lastChild.textContent = 'Not enough data for correlation analysis (need ≥3 queries).';
        }

        // ── 3. A/B vs baseline table ──
        var h3c = document.createElement('h3');
        h3c.textContent = 'A/B Comparison vs Baseline';
        h3c.style.cssText = 'color:#8b949e;font-size:14px;margin:32px 0 12px 0;';
        panel.appendChild(h3c);

        if (statsData.baseline_ab) {
            var ab = statsData.baseline_ab;

            // Info row
            var infoP = document.createElement('p');
            infoP.style.cssText = 'color:#8b949e;font-size:12px;margin-bottom:12px;';
            infoP.textContent = 'Baseline: ' + (baselineRunData ? R.esc(baselineRunData.server_name) : R.esc(ab.baseline_run_id)) +
                ' | Dataset: ' + R.esc(ab.dataset_version || '?');
            panel.appendChild(infoP);

            var table2 = document.createElement('table');
            table2.innerHTML = '<thead><tr>' +
                '<th>Metric</th><th>Delta</th><th>p-value</th><th>Cohen\'s d</th><th>Cliff\'s Delta</th><th>Test</th>' +
                '</tr></thead><tbody></tbody>';
            panel.appendChild(table2);

            var tbody2 = table2.querySelector('tbody');
            allMetrics.forEach(function (m) {
                var delta = ab.delta && ab.delta[m];
                var pval = ab.p_values && ab.p_values[m];
                var cd = ab.cohens_d && ab.cohens_d[m];
                var cliff = ab.cliffs_delta && ab.cliffs_delta[m];
                var test = ab.test_used && ab.test_used[m];

                var tr = document.createElement('tr');
                tr.innerHTML = '<td style="font-weight:600;">' + R.esc(DISPLAY_LABELS[m] || m) + '</td>' +
                    '<td class="metric">' + formatNum(delta) + '</td>' +
                    '<td class="metric">' + formatPval(pval) + '</td>' +
                    '<td class="metric">' + formatNum(cd) + '</td>' +
                    '<td class="metric">' + formatNum(cliff) + '</td>' +
                    '<td>' + R.esc(test || '—') + '</td>';
                tbody2.appendChild(tr);
            });
        } else {
            panel.appendChild(document.createElement('p'));
            panel.lastChild.style.cssText = 'color:#8b949e;padding:16px;';
            panel.lastChild.textContent = 'No A/B baseline comparison available. A grep-glob baseline with the same dataset version is required for auto-detection.';
        }
    }

    // ── Number formatting helpers ──
    function formatNum(v) {
        if (v === null || v === undefined) return '—';
        if (typeof v === 'number') {
            if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
            return v.toFixed(4);
        }
        return String(v);
    }

    function formatPval(v) {
        if (v === null || v === undefined) return '—';
        if (typeof v !== 'number') return String(v);
        if (v < 0.001) return v.toExponential(3);
        return v.toFixed(4);
    }

    // ── Start loading ──
    loadRun();
})();
