/**
 * EcotoxAI — Frontend JavaScript
 * Handles form submission, animated gauge, navigation,
 * Chart.js model comparison visualization, and DYNAMIC
 * metrics loading from the /api/metrics endpoint.
 */

// ══════════════════════════════════════════════
// Global state
// ══════════════════════════════════════════════

let comparisonChart = null;
let plotsLoaded = false;

// Timestamps of the last-known metrics files (for change detection)
let _lastMetricsTimestamps = { csv: null, model_info: null };

// Polling interval handle
let _metricsPollingTimer = null;
const METRICS_POLL_INTERVAL_MS = 30000; // 30 seconds

// ══════════════════════════════════════════════
// Bootstrap
// ══════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initPredictionForm();
    initExampleButton();
    initDynamicMetrics();   // loads metrics + chart + starts polling
    initDynamicPlots();
    initLightbox();
});

// ══════════════════════════════════════════════
// Navigation
// ══════════════════════════════════════════════

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link[data-section]');
    const sections = document.querySelectorAll('.section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('data-section');
            
            // Update active states
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            sections.forEach(s => s.classList.remove('active-section'));
            const target = document.getElementById(targetId);
            if (target) {
                target.classList.add('active-section');
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                if (targetId === 'dashboard') {
                    // Force a fresh metrics check when the user navigates to the dashboard
                    refreshMetrics();
                    initDynamicPlots();
                }
            }
        });
    });
}

// ══════════════════════════════════════════════
// Prediction Form
// ══════════════════════════════════════════════

function initPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handlePrediction();
    });
}

async function handlePrediction() {
    const form = document.getElementById('prediction-form');
    const btn = document.getElementById('predict-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    const resultCard = document.getElementById('result-card');
    const badgeText = document.getElementById('badge-text');
    const badgeEmoji = document.getElementById('badge-emoji');
    const helperText = document.getElementById('result-helper-text');
    
    // Show loading state
    btnText.textContent = 'Predicting...';
    btnLoader.style.display = 'inline';
    btn.disabled = true;
    if (resultCard) resultCard.style.display = 'block';
    if (badgeEmoji) badgeEmoji.textContent = '...';
    if (badgeText) badgeText.textContent = 'Checking toxicity...';
    if (helperText) helperText.textContent = 'The model is analyzing the descriptors and preparing the LC50 prediction.';
    
    try {
        const formData = new FormData(form);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Show results
        displayResult(data);
        
    } catch (error) {
        showError('Connection error. Is the server running?');
    } finally {
        btnText.textContent = 'Predict Toxicity';
        btnLoader.style.display = 'none';
        btn.disabled = false;
    }
}

function displayResult(data) {
    const resultCard = document.getElementById('result-card');
    const gaugeValue = document.getElementById('gauge-value');
    const badgeEmoji = document.getElementById('badge-emoji');
    const badgeText = document.getElementById('badge-text');
    const detailLc50 = document.getElementById('detail-lc50');
    const detailClass = document.getElementById('detail-class');
    const resultDetails = document.getElementById('result-details');
    const helperText = document.getElementById('result-helper-text');
    const badge = document.getElementById('classification-badge');
    
    // Show result card with animation
    resultCard.style.display = 'block';
    resultCard.style.animation = 'none';
    resultCard.offsetHeight; // Trigger reflow
    resultCard.style.animation = 'slideUp 0.5s ease forwards';
    
    // Animate gauge
    const prediction = data.predicted_lc50_neg_log_mol_L;
    animateGauge(prediction);
    
    // Update value
    gaugeValue.textContent = prediction.toFixed(4);
    
    // Update classification badge
    badgeEmoji.textContent = data.toxicity_emoji;
    badgeText.textContent = data.toxicity_classification;
    if (helperText) {
        helperText.textContent = 'Toxicity check complete. Review the predicted LC50, class, and model details below.';
    }
    
    // Color the badge based on toxicity
    const colorMap = {
        'Very High Toxicity': 'rgba(239, 68, 68, 0.2)',
        'High Toxicity': 'rgba(249, 115, 22, 0.2)',
        'Moderate Toxicity': 'rgba(245, 158, 11, 0.2)',
        'Low Toxicity': 'rgba(16, 185, 129, 0.2)',
    };
    const borderMap = {
        'Very High Toxicity': 'rgba(239, 68, 68, 0.5)',
        'High Toxicity': 'rgba(249, 115, 22, 0.5)',
        'Moderate Toxicity': 'rgba(245, 158, 11, 0.5)',
        'Low Toxicity': 'rgba(16, 185, 129, 0.5)',
    };
    
    badge.style.background = colorMap[data.toxicity_classification] || colorMap['Low Toxicity'];
    badge.style.borderColor = borderMap[data.toxicity_classification] || borderMap['Low Toxicity'];
    
    // Update details
    detailLc50.textContent = `${prediction.toFixed(4)} [-LOG(mol/L)]`;
    detailClass.textContent = data.toxicity_classification;
    resultDetails.style.display = 'block';
}

function animateGauge(value) {
    const arc = document.getElementById('gauge-arc');
    const needle = document.getElementById('gauge-needle');
    
    // Normalize value to 0-1 range (assuming LC50 range 0-10)
    const maxVal = 10;
    const normalized = Math.min(Math.max(value / maxVal, 0), 1);
    
    // Arc length
    const totalLength = 251.2; // Approximate arc length
    const targetLength = normalized * totalLength;
    
    // Needle rotation (-90 to 90 degrees)
    const targetRotation = -90 + (normalized * 180);
    
    // Animate
    let currentLength = 0;
    let currentRotation = -90;
    const duration = 1000; // ms
    const startTime = performance.now();
    
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease-out cubic)
        const eased = 1 - Math.pow(1 - progress, 3);
        
        currentLength = eased * targetLength;
        currentRotation = -90 + (eased * (targetRotation + 90));
        
        arc.setAttribute('stroke-dasharray', `${currentLength} ${totalLength}`);
        needle.setAttribute('transform', `rotate(${currentRotation}, 100, 100)`);
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    requestAnimationFrame(animate);
}

function showError(message) {
    const resultCard = document.getElementById('result-card');
    const badgeText = document.getElementById('badge-text');
    const badgeEmoji = document.getElementById('badge-emoji');
    const resultDetails = document.getElementById('result-details');
    const helperText = document.getElementById('result-helper-text');
    
    resultCard.style.display = 'block';
    badgeEmoji.textContent = '❌';
    badgeText.textContent = message;
    if (helperText) {
        helperText.textContent = 'Please check the descriptor values and try again.';
    }
    resultDetails.style.display = 'none';
    
    const badge = document.getElementById('classification-badge');
    badge.style.background = 'rgba(239, 68, 68, 0.15)';
    badge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
}

// ══════════════════════════════════════════════
// Example Data
// ══════════════════════════════════════════════

function initExampleButton() {
    const btn = document.getElementById('example-btn');
    if (!btn) return;
    
    btn.addEventListener('click', () => {
        // Example: a moderately toxic compound from the dataset
        const examples = [
            { CIC0: 1.732, SM1_Dz_Z: 1.393, GATS1i: 0.849, NdsCH: 1, NdssC: 0, MLOGP: 2.246 },
            { CIC0: 3.024, SM1_Dz_Z: 2.091, GATS1i: 1.127, NdsCH: 0, NdssC: 0, MLOGP: 4.680 },
            { CIC0: 2.441, SM1_Dz_Z: 1.569, GATS1i: 0.928, NdsCH: 2, NdssC: 1, MLOGP: 1.340 },
            { CIC0: 1.282, SM1_Dz_Z: 0.846, GATS1i: 1.058, NdsCH: 0, NdssC: 0, MLOGP: 0.700 },
        ];
        
        const example = examples[Math.floor(Math.random() * examples.length)];
        
        document.getElementById('CIC0').value = example.CIC0;
        document.getElementById('SM1_Dz_Z').value = example.SM1_Dz_Z;
        document.getElementById('GATS1i').value = example.GATS1i;
        document.getElementById('NdsCH').value = example.NdsCH;
        document.getElementById('NdssC').value = example.NdssC;
        document.getElementById('MLOGP').value = example.MLOGP;
        
        // Visual feedback
        btn.querySelector('.btn-text').textContent = 'Loaded! ✓';
        setTimeout(() => {
            btn.querySelector('.btn-text').textContent = 'Load Example';
        }, 1500);
    });
}

// ══════════════════════════════════════════════
// Dynamic Metrics (Detailed Metrics + Chart)
// ══════════════════════════════════════════════

/**
 * Initialise the dynamic metrics system:
 *  1. Fetch latest metrics from /api/metrics
 *  2. Populate the table & chart
 *  3. Wire up the manual refresh button
 *  4. Start the background polling timer
 */
function initDynamicMetrics() {
    // Wire up the manual refresh button
    const refreshBtn = document.getElementById('refresh-metrics-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => refreshMetrics());
    }

    // Initial load
    refreshMetrics();

    // Start background polling
    startMetricsPolling();
}

/**
 * Fetch the latest metrics from the server and update the UI.
 * Optionally pass `force = true` to skip the timestamp check.
 */
async function refreshMetrics(force = true) {
    const refreshBtn = document.getElementById('refresh-metrics-btn');

    try {
        if (!force) {
            // Lightweight check: only fetch timestamps to see if anything changed
            const checkResp = await fetch('/api/metrics/check');
            if (!checkResp.ok) return;
            const ts = await checkResp.json();
            if (ts.csv === _lastMetricsTimestamps.csv &&
                ts.model_info === _lastMetricsTimestamps.model_info) {
                return; // nothing changed — skip the full fetch
            }
        }

        // Spin the refresh icon
        refreshBtn && refreshBtn.classList.add('spinning');

        const response = await fetch('/api/metrics');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        // Remember timestamps for next poll
        _lastMetricsTimestamps = data.timestamps || {};

        // Update global comparisonData for the chart
        window.comparisonData = data.comparison || [];

        // Toggle empty-state vs content
        const emptyState = document.getElementById('dashboard-empty-state');
        const chartWrapper = document.getElementById('chart-card-wrapper');
        const metricsWrapper = document.getElementById('metrics-card-wrapper');

        if (!data.has_data) {
            if (emptyState) emptyState.style.display = '';
            if (chartWrapper) chartWrapper.style.display = 'none';
            if (metricsWrapper) metricsWrapper.style.display = 'none';
            return;
        }

        if (emptyState) emptyState.style.display = 'none';
        if (chartWrapper) chartWrapper.style.display = '';
        if (metricsWrapper) metricsWrapper.style.display = '';

        // Rebuild the table
        buildMetricsTable(data.comparison);

        // Rebuild the chart
        buildComparisonChart(data.comparison);

        // Update the "last updated" label
        updateMetricsTimestamp(data.timestamps);

        // Show a brief "updated" flash on the status bar
        showMetricsStatusFlash('✅ Metrics synced with latest training run');

    } catch (err) {
        console.error('Failed to refresh metrics:', err);
        showMetricsStatusFlash('⚠️ Could not load metrics — retrying…', true);
    } finally {
        refreshBtn && refreshBtn.classList.remove('spinning');
    }
}

/**
 * Build (or rebuild) the metrics table from an array of model objects.
 */
function buildMetricsTable(comparison) {
    const tbody = document.getElementById('metrics-table-body');
    if (!tbody) return;

    if (!comparison || comparison.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align:center; padding:2rem; color:var(--text-muted);">
                    No metrics data available. Run <code>python train.py</code> to generate results.
                </td>
            </tr>`;
        return;
    }

    // Dynamically detect all available column keys (beyond the standard ones)
    const standardKeys = ['Model', 'R²', 'RMSE', 'MAE', 'MAPE (%)', 'CV R² (mean)', 'CV R² (std)'];

    tbody.innerHTML = comparison.map((model, index) => {
        const rank = index + 1;
        const rankEmoji = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : rank;
        const topClass = rank <= 3 ? 'top-model' : '';

        const r2 = formatMetric(model['R²'], 4);
        const rmse = formatMetric(model['RMSE'], 4);
        const mae = formatMetric(model['MAE'], 4);
        const mape = formatMetric(model['MAPE (%)'], 1, '%');
        const cvMean = formatMetric(model['CV R² (mean)'], 4);
        const cvStd = formatMetric(model['CV R² (std)'], 4);

        return `
            <tr class="${topClass}" style="animation: fadeSlideIn 0.3s ease ${index * 0.04}s both;">
                <td>${rankEmoji}</td>
                <td class="model-name">${model.Model || 'N/A'}</td>
                <td>${r2}</td>
                <td>${rmse}</td>
                <td>${mae}</td>
                <td>${mape}</td>
                <td>${cvMean}</td>
                <td>${cvStd}</td>
            </tr>`;
    }).join('');
}

/**
 * Format a metric value for display.
 */
function formatMetric(value, decimals, suffix = '') {
    if (value === undefined || value === null || value === '') return '—';
    const num = Number(value);
    if (isNaN(num)) return String(value);
    return num.toFixed(decimals) + suffix;
}

/**
 * Update the "last updated" timestamp near the refresh button.
 */
function updateMetricsTimestamp(timestamps) {
    const el = document.getElementById('metrics-last-updated');
    if (!el) return;

    const csvTs = timestamps && timestamps.csv;
    if (csvTs) {
        const date = new Date(csvTs * 1000);
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const dateStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
        el.textContent = `Last trained: ${dateStr} ${timeStr}`;
        el.title = date.toISOString();
    } else {
        el.textContent = '';
    }
}

/**
 * Show a brief status flash in the metrics card.
 */
function showMetricsStatusFlash(message, isError = false) {
    const bar = document.getElementById('metrics-status-bar');
    if (!bar) return;

    bar.textContent = message;
    bar.className = isError ? 'metrics-status-bar error' : 'metrics-status-bar success';
    bar.style.display = 'block';

    // Auto-hide after 4 seconds
    clearTimeout(bar._hideTimer);
    bar._hideTimer = setTimeout(() => {
        bar.style.display = 'none';
    }, 4000);
}

/**
 * Start polling the server for metrics changes every METRICS_POLL_INTERVAL_MS.
 * Uses the lightweight /api/metrics/check endpoint first to avoid
 * unnecessary heavy fetches.
 */
function startMetricsPolling() {
    if (_metricsPollingTimer) clearInterval(_metricsPollingTimer);
    _metricsPollingTimer = setInterval(() => {
        refreshMetrics(false);  // false = check timestamps first
    }, METRICS_POLL_INTERVAL_MS);
}

// ══════════════════════════════════════════════
// Model Comparison Chart
// ══════════════════════════════════════════════

/**
 * Legacy wrapper preserved for any callers that used the old name.
 */
function initComparisonChart() {
    if (window.comparisonData && window.comparisonData.length > 0) {
        buildComparisonChart(window.comparisonData);
    }
}

/**
 * Build (or rebuild) the Chart.js bar chart from comparison data.
 */
function buildComparisonChart(comparisonData) {
    const canvas = document.getElementById('comparison-chart');
    if (!canvas || !comparisonData || comparisonData.length === 0) return;
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    
    const modelNames = comparisonData.map(m => m.Model || 'Unknown');
    const r2Scores = comparisonData.map(m => m['R²'] || 0);
    const rmseScores = comparisonData.map(m => m.RMSE || 0);
    const maeScores = comparisonData.map(m => m.MAE || 0);
    
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [
                {
                    label: 'R² Score',
                    data: r2Scores,
                    backgroundColor: 'rgba(99, 102, 241, 0.6)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: 'RMSE',
                    data: rmseScores,
                    backgroundColor: 'rgba(6, 182, 212, 0.6)',
                    borderColor: 'rgba(6, 182, 212, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: 'MAE',
                    data: maeScores,
                    backgroundColor: 'rgba(139, 92, 246, 0.6)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12 },
                        padding: 16,
                        usePointStyle: true,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    titleFont: { family: 'Inter', weight: '600' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#64748b',
                        font: { family: 'Inter', size: 11 },
                        maxRotation: 45,
                    },
                    grid: { display: false },
                },
                y: {
                    ticks: {
                        color: '#64748b',
                        font: { family: 'JetBrains Mono', size: 11 },
                    },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart',
            }
        }
    });
}

// ══════════════════════════════════════════════
// Dynamic Plot Gallery
// ══════════════════════════════════════════════

function initDynamicPlots() {
    const grid = document.getElementById('dynamic-plots-grid');
    if (!grid) return;

    // Only auto-load once; manual refresh always reloads
    if (plotsLoaded) return;
    loadPlots();
}

async function loadPlots() {
    const grid = document.getElementById('dynamic-plots-grid');
    const countEl = document.getElementById('plots-count');
    const refreshBtn = document.getElementById('refresh-plots-btn');
    if (!grid) return;

    // Show loading state
    refreshBtn && refreshBtn.classList.add('spinning');
    if (!plotsLoaded) {
        grid.innerHTML = `
            <div class="plot-skeleton"><div class="skeleton-shimmer"></div></div>
            <div class="plot-skeleton"><div class="skeleton-shimmer"></div></div>
            <div class="plot-skeleton"><div class="skeleton-shimmer"></div></div>
        `;
    }

    try {
        const response = await fetch('/api/plots');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        if (!data.plots || data.plots.length === 0) {
            grid.innerHTML = `
                <div class="plots-empty-state">
                    <span class="empty-icon">📊</span>
                    <h4>No Plots Available</h4>
                    <p>Run <code>python train.py</code> to generate training visualizations.</p>
                </div>
            `;
            if (countEl) countEl.textContent = '';
            plotsLoaded = true;
            return;
        }

        // Update count badge
        if (countEl) {
            countEl.textContent = `${data.count} visualization${data.count !== 1 ? 's' : ''} found`;
        }

        // Cache-bust timestamp
        const cacheBust = Date.now();

        // Build plot cards
        grid.innerHTML = data.plots.map((plot, index) => `
            <div class="dyn-plot-item" style="animation-delay: ${index * 0.05}s">
                <div class="dyn-plot-img-container" data-src="${plot.url}" data-title="${plot.title}">
                    <img
                        src="${plot.url}?t=${cacheBust}"
                        alt="${plot.title}"
                        class="dyn-plot-img"
                        loading="lazy"
                        onerror="this.parentElement.classList.add('img-error'); this.style.display='none';"
                    >
                    <div class="img-error-placeholder">
                        <span class="error-icon">🖼️</span>
                        <span>Image unavailable</span>
                    </div>
                    <div class="plot-zoom-hint">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>
                        Click to enlarge
                    </div>
                </div>
                <div class="dyn-plot-title">${plot.title}</div>
            </div>
        `).join('');

        plotsLoaded = true;

    } catch (error) {
        console.error('Failed to load plots:', error);
        grid.innerHTML = `
            <div class="plots-empty-state plots-error-state">
                <span class="empty-icon">⚠️</span>
                <h4>Failed to Load Plots</h4>
                <p>Could not fetch plot data from the server. Please try refreshing.</p>
            </div>
        `;
    } finally {
        refreshBtn && refreshBtn.classList.remove('spinning');
    }
}

// ══════════════════════════════════════════════
// Lightbox
// ══════════════════════════════════════════════

function initLightbox() {
    const overlay = document.getElementById('lightbox-overlay');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxTitle = document.getElementById('lightbox-title');
    const closeBtn = document.getElementById('lightbox-close');
    const grid = document.getElementById('dynamic-plots-grid');
    if (!overlay || !grid) return;

    // Event delegation: click on any plot image container
    grid.addEventListener('click', (e) => {
        const container = e.target.closest('.dyn-plot-img-container');
        if (!container || container.classList.contains('img-error')) return;

        const src = container.dataset.src;
        const title = container.dataset.title;
        if (!src) return;

        lightboxImg.src = src + '?t=' + Date.now();
        lightboxImg.alt = title || '';
        lightboxTitle.textContent = title || '';
        overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    });

    // Close handlers
    function closeLightbox() {
        overlay.classList.remove('active');
        document.body.style.overflow = '';
        // Clear src after transition to free memory
        setTimeout(() => { lightboxImg.src = ''; }, 300);
    }

    closeBtn.addEventListener('click', closeLightbox);
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) closeLightbox();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && overlay.classList.contains('active')) {
            closeLightbox();
        }
    });

    // Refresh button handler (plots)
    const refreshBtn = document.getElementById('refresh-plots-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            plotsLoaded = false;
            loadPlots();
        });
    }
}
