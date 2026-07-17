/**
 * EcoTox-AI — Frontend JavaScript
 * Handles form submission, animated gauge, navigation,
 * and Chart.js model comparison visualization.
 */

// ══════════════════════════════════════════════
// Navigation
// ══════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initPredictionForm();
    initExampleButton();
    initComparisonChart();
    initDynamicPlots();
    initLightbox();
});

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
                    initComparisonChart();
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
    
    // Show loading state
    btnText.textContent = 'Predicting...';
    btnLoader.style.display = 'inline';
    btn.disabled = true;
    
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
    
    resultCard.style.display = 'block';
    badgeEmoji.textContent = '❌';
    badgeText.textContent = message;
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
// Model Comparison Chart
// ══════════════════════════════════════════════

let comparisonChart = null;

function initComparisonChart() {
    const canvas = document.getElementById('comparison-chart');
    if (!canvas || !window.comparisonData || comparisonData.length === 0) return;
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    
    const modelNames = comparisonData.map(m => m.Model || 'Unknown');
    const r2Scores = comparisonData.map(m => m['R²'] || 0);
    const rmseScores = comparisonData.map(m => m.RMSE || 0);
    const maeScores = comparisonData.map(m => m.MAE || 0);
    
    // Generate colors
    const colors = modelNames.map((_, i) => {
        const hue = (i * 360 / modelNames.length + 220) % 360;
        return `hsla(${hue}, 70%, 60%, 0.8)`;
    });
    
    const borderColors = modelNames.map((_, i) => {
        const hue = (i * 360 / modelNames.length + 220) % 360;
        return `hsla(${hue}, 70%, 60%, 1)`;
    });
    
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

let plotsLoaded = false;

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

    // Refresh button handler
    const refreshBtn = document.getElementById('refresh-plots-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            plotsLoaded = false;
            loadPlots();
        });
    }
}
