/**
 * ExoDip Frontend Application Logic & Reactive State Controller.
 * Handles client-side routing, API integration, Plotly.js charts,
 * drag-and-drop file ingestion, and session history.
 */

// ==========================================
// 1. Application State
// ==========================================
const state = {
    page: 'Home',
    selectedFile: null,
    activeResult: null,
    history: []
};

// Load persistent history from browser localStorage
try {
    const saved = localStorage.getItem('exodip_history');
    if (saved) {
        state.history = JSON.parse(saved);
    }
} catch (e) {
    console.warn('Could not load history from localStorage:', e);
    state.history = [];
}

// ==========================================
// 2. Navigation Controller & Mobile Menu
// ==========================================
function toggleMobileMenu() {
    const drawer = document.getElementById('mobile-drawer');
    const btn = document.getElementById('mobile-menu-btn');
    const backdrop = document.getElementById('mobile-backdrop');
    if (!drawer) return;

    const isOpen = drawer.classList.contains('open');
    if (isOpen) {
        closeMobileMenu();
    } else {
        drawer.classList.add('open');
        if (btn) btn.classList.add('active');
        if (backdrop) backdrop.classList.add('open');
        document.body.style.overflow = 'hidden';
    }
}

function closeMobileMenu() {
    const drawer = document.getElementById('mobile-drawer');
    const btn = document.getElementById('mobile-menu-btn');
    const backdrop = document.getElementById('mobile-backdrop');
    if (drawer) drawer.classList.remove('open');
    if (btn) btn.classList.remove('active');
    if (backdrop) backdrop.classList.remove('open');
    document.body.style.overflow = '';
}

function navigateTo(pageName) {
    state.page = pageName;

    // Update navbar active state (desktop)
    document.querySelectorAll('.nav-item').forEach(el => {
        if (el.getAttribute('data-page') === pageName) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });

    // Update mobile drawer active state
    document.querySelectorAll('.mobile-nav-item').forEach(el => {
        if (el.getAttribute('data-page') === pageName) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });

    // Close mobile menu if open
    closeMobileMenu();

    // Toggle view visibility
    document.querySelectorAll('.view-page').forEach(el => el.style.display = 'none');
    const target = document.getElementById(`view-${pageName.toLowerCase()}`);
    if (target) {
        target.style.display = 'block';
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Page-specific lifecycle hooks
    if (pageName === 'Results') {
        renderResultsView();
    } else if (pageName === 'History') {
        renderHistoryView();
    }
}

// ==========================================
// 3. Tab Switching Controllers
// ==========================================
function switchAnalyzeTab(tabId) {
    const parent = document.querySelector('#view-analyze');
    parent.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    parent.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    if (tabId === 'upload') {
        parent.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('analyze-tab-upload').classList.add('active');
    } else {
        parent.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('analyze-tab-test-row').classList.add('active');
    }
}

function switchResultTab(tabKey) {
    const parent = document.querySelector('#view-results');
    parent.querySelectorAll('.tab-nav .tab-btn').forEach(b => b.classList.remove('active'));
    parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    const tabMap = {
        'full': { index: 0, id: 'res-tab-full' },
        'zoom': { index: 1, id: 'res-tab-zoom' },
        'models': { index: 2, id: 'res-tab-models' },
        'features': { index: 3, id: 'res-tab-features' }
    };

    const target = tabMap[tabKey];
    if (target) {
        parent.querySelectorAll('.tab-nav .tab-btn')[target.index].classList.add('active');
        document.getElementById(target.id).classList.add('active');

        // Trigger Plotly relayout to ensure proper dimensions on tab reveal
        if (tabKey === 'full' && document.getElementById('chart-full')) {
            Plotly.Plots.resize('chart-full');
        } else if (tabKey === 'zoom' && document.getElementById('chart-zoom')) {
            Plotly.Plots.resize('chart-zoom');
        } else if (tabKey === 'models' && document.getElementById('chart-probs')) {
            Plotly.Plots.resize('chart-probs');
        }
    }
}

function switchDocsTab(tabKey) {
    const parent = document.querySelector('#view-docs');
    parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    const tabMap = {
        'start': { index: 0, id: 'docs-tab-start' },
        'formats': { index: 1, id: 'docs-tab-formats' },
        'pipeline': { index: 2, id: 'docs-tab-pipeline' },
        'results': { index: 3, id: 'docs-tab-results' },
        'limitations': { index: 4, id: 'docs-tab-limitations' },
        'faq': { index: 5, id: 'docs-tab-faq' }
    };

    const target = tabMap[tabKey];
    if (target) {
        parent.querySelectorAll('.tab-btn')[target.index].classList.add('active');
        document.getElementById(target.id).classList.add('active');
    }
}

// ==========================================
// 4. File Drag & Drop Handling
// ==========================================
const dropZone = document.getElementById('drop-zone');
if (dropZone) {
    ['dragenter', 'dragover'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelected(files[0]);
        }
    });
}

function handleFileSelected(file) {
    if (!file) return;
    state.selectedFile = file;

    const previewCard = document.getElementById('file-preview-card');
    const nameEl = document.getElementById('file-preview-name');
    const sizeEl = document.getElementById('file-preview-size');
    const formatEl = document.getElementById('file-preview-format');

    const sizeKb = file.size / 1024.0;
    const sizeStr = sizeKb < 1024 ? `${sizeKb.toFixed(1)} KB` : `${(sizeKb / 1024.0).toFixed(2)} MB`;
    const ext = file.name.split('.').pop().toUpperCase();

    nameEl.textContent = file.name;
    sizeEl.textContent = sizeStr;
    formatEl.textContent = ext;

    previewCard.style.display = 'block';
}

// ==========================================
// 5. Analysis Execution & API Calls
// ==========================================
async function animateProgressSteps() {
    const steps = ['step-1', 'step-2', 'step-3', 'step-4', 'step-5'];
    steps.forEach(id => {
        const el = document.getElementById(id);
        el.className = 'loading-step';
        el.querySelector('span').textContent = '○';
    });

    document.getElementById('analysis-loading-banner').style.display = 'block';

    for (let i = 0; i < steps.length; i++) {
        const el = document.getElementById(steps[i]);
        el.className = 'loading-step active';
        el.querySelector('span').textContent = '▶';
        await new Promise(r => setTimeout(r, 220));
        el.className = 'loading-step done';
        el.querySelector('span').textContent = '✓';
    }
}

async function startFileAnalysis() {
    if (!state.selectedFile) {
        alert('Please select a file to screen.');
        return;
    }

    const formData = new FormData();
    formData.append('file', state.selectedFile);

    const progressPromise = animateProgressSteps();

    try {
        const res = await fetch(BACKEND_URL + '/api/analyze/file', {
            method: 'POST',
            body: formData
        });

        await progressPromise;

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Analysis failed.' }));
            throw new Error(err.detail || 'Failed to screen light curve.');
        }

        const data = await res.json();
        saveAndDisplayResult(data);

    } catch (err) {
        document.getElementById('analysis-loading-banner').style.display = 'none';
        alert('Error: ' + err.message);
    }
}

async function startTestRowAnalysis() {
    const rowInput = document.getElementById('test-row-id');
    const rowId = parseInt(rowInput.value, 10);

    if (isNaN(rowId) || rowId < 0) {
        alert('Please provide a valid non-negative row ID.');
        return;
    }

    const progressPromise = animateProgressSteps();

    try {
        const res = await fetch(BACKEND_URL + '/api/analyze/test-row', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ row_id: rowId })
        });

        await progressPromise;

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Failed to screen test row.' }));
            throw new Error(err.detail || 'Failed to screen test row.');
        }

        const data = await res.json();
        saveAndDisplayResult(data);

    } catch (err) {
        document.getElementById('analysis-loading-banner').style.display = 'none';
        alert('Error: ' + err.message);
    }
}

function saveAndDisplayResult(data) {
    document.getElementById('analysis-loading-banner').style.display = 'none';

    // Set active result
    state.activeResult = data;

    // Add to history
    state.history.unshift({
        id: data.id,
        source: data.source,
        prediction: data.prediction,
        probability: data.probability,
        confidence: data.confidence,
        model: data.model,
        timestamp: data.timestamp,
        data: data
    });

    // Persist to localStorage
    try {
        localStorage.setItem('exodip_history', JSON.stringify(state.history));
    } catch (e) {
        console.warn('Could not save to localStorage:', e);
    }

    navigateTo('Results');
}

// ==========================================
// 6. Results Rendering & Visualizations
// ==========================================
function renderResultsView() {
    const emptyEl = document.getElementById('results-empty');
    const contentEl = document.getElementById('results-content');

    if (!state.activeResult) {
        emptyEl.style.display = 'block';
        contentEl.style.display = 'none';
        return;
    }

    emptyEl.style.display = 'none';
    contentEl.style.display = 'block';

    const r = state.activeResult;
    const isPlanet = r.prediction === 'Planet';

    // Source line
    document.getElementById('res-source-line').textContent = `${r.source} · Analyzed on ${r.timestamp}`;

    // Verdict Badge
    const badgeContainer = document.getElementById('res-badge-container');
    const verdictDesc = document.getElementById('res-verdict-desc');

    if (isPlanet) {
        badgeContainer.innerHTML = "<div class='candidate-badge-candidate'>🪐 EXOPLANET CANDIDATE</div>";
        verdictDesc.textContent = "Candidate status indicates periodic transit-like signal characteristics warranting further astronomical follow-up.";
    } else {
        badgeContainer.innerHTML = "<div class='candidate-badge-non'>⚪ NON-CANDIDATE</div>";
        verdictDesc.textContent = "Signal does not exhibit sufficient periodic transit depth, SNR, or classifier confidence to qualify as a candidate.";
    }

    // Confidence
    const confVal = r.confidence != null ? r.confidence : 0;
    document.getElementById('res-conf-value').textContent = `${confVal.toFixed(1)}%`;
    document.getElementById('res-conf-bar').style.width = `${Math.min(Math.max(confVal, 0), 100)}%`;

    // Detection Telemetry
    document.getElementById('res-tele-model').textContent = `⭐ ${r.model}`;
    document.getElementById('res-tele-dips').textContent = r.chart_data ? r.chart_data.dips.length : 0;
    document.getElementById('res-tele-noise').textContent = r.features ? r.features.noise_level.toPrecision(4) : '0';
    document.getElementById('res-tele-neg').textContent = r.features ? r.features.negative_ratio.toPrecision(4) : '0';

    // Model Agreement calculation
    const modelProbs = r.model_probabilities || {};
    const totalVotes = Object.keys(modelProbs).length || 1;
    const planetVotes = Object.values(modelProbs).filter(p => p >= 0.5).length;
    const agreeEl = document.getElementById('res-tele-agree');

    if (planetVotes === totalVotes || planetVotes === 0) {
        agreeEl.textContent = 'Unanimous (All models agree)';
        agreeEl.style.color = '#34d399';
    } else if (isPlanet) {
        agreeEl.textContent = `Consensus (${planetVotes}/${totalVotes} models agree)`;
        agreeEl.style.color = '#38bdf8';
    } else {
        agreeEl.textContent = `Consensus (${totalVotes - planetVotes}/${totalVotes} models agree)`;
        agreeEl.style.color = '#38bdf8';
    }

    // Signal Metrics Strip
    const m = r.metrics || {};
    const blsP = m.bls_period_days;
    const blsDepth = m.bls_transit_depth != null ? m.bls_transit_depth : m.transit_depth;
    const blsDur = m.bls_duration_hours;
    const blsSnr = m.bls_transit_snr;

    document.getElementById('res-m-period').textContent = (blsP != null && blsP > 0) ? `${blsP.toFixed(2)} d` : 'N/A';
    document.getElementById('res-m-depth').textContent = (blsDepth != null && blsDepth > 0) ? `${(blsDepth < 1.0 ? blsDepth * 100 : blsDepth).toFixed(3)}%` : '0.000%';
    document.getElementById('res-m-duration').textContent = (blsDur != null && blsDur > 0) ? `${blsDur.toFixed(1)} h` : (r.features ? `${r.features.avg_transit_duration.toFixed(1)} pts` : 'N/A');
    document.getElementById('res-m-snr').textContent = (blsSnr != null && blsSnr > 0) ? blsSnr.toFixed(1) : '0.0';
    document.getElementById('res-m-groups').textContent = r.chart_data ? r.chart_data.dips.length : 0;
    document.getElementById('res-m-model').textContent = r.model;

    // Fill Features tab
    const f = r.features || {};
    document.getElementById('feat-mean').textContent = f.mean != null ? f.mean.toPrecision(4) : '0.0';
    document.getElementById('feat-median').textContent = f.median != null ? f.median.toPrecision(4) : '0.0';
    document.getElementById('feat-var').textContent = f.variance != null ? f.variance.toPrecision(4) : '0.0';
    document.getElementById('feat-skew').textContent = f.skewness != null ? f.skewness.toPrecision(4) : '0.0';
    document.getElementById('feat-kurt').textContent = f.kurtosis != null ? f.kurtosis.toPrecision(4) : '0.0';

    document.getElementById('feat-energy').textContent = f.signal_energy != null ? f.signal_energy.toPrecision(4) : '0.0';
    document.getElementById('feat-entropy').textContent = f.spectral_entropy != null ? f.spectral_entropy.toPrecision(4) : '0.0';
    document.getElementById('feat-neg').textContent = f.negative_ratio != null ? f.negative_ratio.toPrecision(4) : '0.0';
    document.getElementById('feat-noise').textContent = f.noise_level != null ? f.noise_level.toPrecision(4) : '0.0';

    document.getElementById('feat-depth').textContent = f.avg_transit_depth != null ? f.avg_transit_depth.toPrecision(4) : '0.0';
    document.getElementById('feat-dur').textContent = f.avg_transit_duration != null ? `${f.avg_transit_duration.toFixed(1)} pts` : '0.0';
    document.getElementById('feat-dips').textContent = f.num_dips != null ? f.num_dips : '0';
    document.getElementById('feat-min').textContent = f.min_flux != null ? f.min_flux.toPrecision(4) : '0.0';

    document.getElementById('feat-bls-p').textContent = (blsP != null && blsP > 0) ? `${blsP.toFixed(2)} d` : 'N/A';
    document.getElementById('feat-bls-dur').textContent = (blsDur != null && blsDur > 0) ? `${blsDur.toFixed(1)} h` : 'N/A';
    document.getElementById('feat-bls-snr').textContent = (blsSnr != null && blsSnr > 0) ? blsSnr.toFixed(1) : '0.0';
    document.getElementById('feat-bls-pow').textContent = m.bls_power != null ? m.bls_power.toPrecision(4) : '0.0';

    // Render Models Table
    renderModelsTable(r);

    // Render Plotly Charts
    renderCharts(r);
}

function renderModelsTable(r) {
    const tbody = document.getElementById('res-models-tbody');
    tbody.innerHTML = '';
    const scores = r.model_scores || {};
    const probs = r.model_probabilities || {};

    const rows = [];
    for (const [mname, sc] of Object.entries(scores)) {
        const prob = probs[mname] || 0.0;
        const isSelected = mname === r.model;
        const verdict = (isSelected ? r.prediction === 'Planet' : prob >= 0.5) ? '🪐 Candidate' : '⚪ Non-Candidate';
        const dispProb = (isSelected && r.prediction === 'Planet') ? (r.probability * 100) : (prob * 100);

        rows.push({
            modelName: (isSelected ? '⭐ ' : '') + mname,
            acc: sc.Accuracy != null ? sc.Accuracy.toFixed(4) : '-',
            prec: sc.Precision != null ? sc.Precision.toFixed(4) : '-',
            rec: sc.Recall != null ? sc.Recall.toFixed(4) : '-',
            f1: sc['F1 Score'] != null ? sc['F1 Score'].toFixed(4) : '-',
            auc: sc['ROC-AUC'] != null ? sc['ROC-AUC'].toFixed(4) : '-',
            comp: sc.Composite != null ? sc.Composite.toFixed(4) : '0',
            compNum: sc.Composite || 0,
            verdict: verdict,
            prob: `${dispProb.toFixed(1)}%`
        });
    }

    rows.sort((a, b) => b.compNum - a.compNum);

    rows.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><b>${row.modelName}</b></td>
            <td>${row.acc}</td>
            <td>${row.prec}</td>
            <td>${row.rec}</td>
            <td>${row.f1}</td>
            <td>${row.auc}</td>
            <td><b>${row.comp}</b></td>
            <td>${row.verdict}</td>
            <td style="color:#38bdf8; font-weight:600;">${row.prob}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderCharts(r) {
    if (!r.chart_data || !window.Plotly) return;

    const cd = r.chart_data;
    const flux = cd.flux || [];
    const smooth = cd.smooth || [];
    const dips = cd.dips || [];

    // --- Chart 1: Full Light Curve ---
    const fullTraces = [
        {
            y: flux,
            type: 'scatter',
            mode: 'lines',
            name: 'Observed Flux',
            line: { color: '#818cf8', width: 1 },
            opacity: 0.65
        },
        {
            y: smooth,
            type: 'scatter',
            mode: 'lines',
            name: 'Smoothed Trend',
            line: { color: '#38bdf8', width: 2 }
        }
    ];

    if (dips.length > 0) {
        fullTraces.push({
            x: dips,
            y: dips.map(i => smooth[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Detected Dips',
            marker: { color: '#f43f5e', size: 5, symbol: 'circle' }
        });
    }

    const fullLayout = {
        title: 'Light Curve with Detected Transit Dips',
        template: 'plotly_dark',
        paper_bgcolor: '#03050a',
        plot_bgcolor: '#03050a',
        height: 400,
        margin: { l: 45, r: 15, t: 42, b: 40 },
        legend: { orientation: 'h', y: 1.05 },
        xaxis: { title: 'Time / Sample Index', gridcolor: '#1b2a42', zeroline: false },
        yaxis: { title: 'Normalized Relative Flux', gridcolor: '#1b2a42', zeroline: false },
        font: { family: 'Inter, sans-serif' }
    };

    Plotly.newPlot('chart-full', fullTraces, fullLayout, { responsive: true, displayModeBar: false });

    // --- Chart 2: Zoomed Transit View ---
    const z = cd.zoom || {};
    const start = z.start || 0;
    const end = z.end || Math.min(flux.length, 150);
    const xRange = Array.from({ length: end - start }, (_, i) => start + i);

    const zoomTraces = [
        {
            x: xRange,
            y: flux.slice(start, end),
            type: 'scatter',
            mode: 'markers+lines',
            name: 'Observed Data Points',
            line: { color: '#818cf8', width: 1 },
            marker: { size: 4, color: '#818cf8' },
            opacity: 0.6
        },
        {
            x: xRange,
            y: smooth.slice(start, end),
            type: 'scatter',
            mode: 'lines',
            name: 'Smoothed Profile',
            line: { color: '#38bdf8', width: 2.5 }
        }
    ];

    if (z.local_dips && z.local_dips.length > 0) {
        zoomTraces.push({
            x: z.local_dips,
            y: z.local_dips.map(i => smooth[i]),
            type: 'scatter',
            mode: 'markers',
            name: 'Transit Ingress/Egress',
            marker: { color: '#f43f5e', size: 7, symbol: 'triangle-down' }
        });
    }

    const zoomTitle = z.has_deepest_dip
        ? `Deepest Transit Dip (Centered at Sample #${z.deepest_idx})`
        : 'Transit View (No Significant Dips Detected)';

    const zoomLayout = {
        title: zoomTitle,
        template: 'plotly_dark',
        paper_bgcolor: '#03050a',
        plot_bgcolor: '#03050a',
        height: 380,
        margin: { l: 45, r: 20, t: 40, b: 40 },
        legend: { orientation: 'h', y: 1.05 },
        xaxis: { title: 'Sample Index', gridcolor: '#1b2a42', zeroline: false },
        yaxis: { title: 'Normalized Relative Flux', gridcolor: '#1b2a42', zeroline: false },
        font: { family: 'Inter, sans-serif' }
    };

    Plotly.newPlot('chart-zoom', zoomTraces, zoomLayout, { responsive: true, displayModeBar: false });

    // --- Chart 3: Model Probabilities Bar Chart ---
    const probs = r.model_probabilities || {};
    const modelNames = Object.keys(probs);
    const probValues = modelNames.map(name => {
        let p = probs[name];
        if (name === r.model && r.prediction === 'Planet') {
            p = r.probability;
        }
        return Math.round(p * 1000) / 10;
    });

    const probTraces = [
        {
            x: modelNames,
            y: probValues,
            type: 'bar',
            marker: { color: '#818cf8', opacity: 0.9 },
            text: probValues.map(v => `${v}%`),
            textposition: 'auto'
        }
    ];

    const probLayout = {
        template: 'plotly_dark',
        paper_bgcolor: '#03050a',
        plot_bgcolor: '#03050a',
        height: 250,
        margin: { l: 40, r: 20, t: 20, b: 45 },
        xaxis: { gridcolor: '#1b2a42' },
        yaxis: { title: 'Probability (%)', range: [0, 100], gridcolor: '#1b2a42' },
        font: { family: 'Inter, sans-serif' }
    };

    Plotly.newPlot('chart-probs', probTraces, probLayout, { responsive: true, displayModeBar: false });
}

// ==========================================
// 7. Session History Controller
// ==========================================
function renderHistoryView() {
    const total = state.history.length;
    const planets = state.history.filter(h => h.prediction === 'Planet').length;
    const rate = total > 0 ? (planets / total * 100.0).toFixed(1) : '0.0';

    document.getElementById('hist-total').textContent = total;
    document.getElementById('hist-planets').textContent = planets;
    document.getElementById('hist-rate').textContent = `${rate}%`;

    const tableContainer = document.getElementById('history-table-container');
    const emptyEl = document.getElementById('history-empty');
    const tbody = document.getElementById('history-tbody');

    if (total === 0) {
        tableContainer.style.display = 'none';
        emptyEl.style.display = 'block';
        return;
    }

    tableContainer.style.display = 'block';
    emptyEl.style.display = 'none';
    tbody.innerHTML = '';

    state.history.forEach((item, idx) => {
        const tr = document.createElement('tr');
        const verdictText = item.prediction === 'Planet' ? '🪐 Candidate' : '⚪ Non-Candidate';
        const confText = item.confidence != null ? `${item.confidence.toFixed(1)}%` : '-';

        tr.innerHTML = `
            <td><b>${item.source}</b></td>
            <td>${verdictText}</td>
            <td style="color:#38bdf8; font-weight:600;">${confText}</td>
            <td>${item.model || 'XGBoost'}</td>
            <td class="muted">${item.timestamp}</td>
            <td><button class="btn btn-primary" style="min-height:2rem; padding:0 0.8rem; font-size:0.85rem;" onclick="openHistoryItem(${idx})">View</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function openHistoryItem(index) {
    if (state.history[index]) {
        state.activeResult = state.history[index].data;
        navigateTo('Results');
    }
}

function clearHistory() {
    if (confirm('Are you sure you want to clear your active session history?')) {
        state.history = [];
        try {
            localStorage.removeItem('exodip_history');
        } catch (e) {}
        renderHistoryView();
    }
}

// ==========================================
// 8. Download Report Handler
// ==========================================
function downloadReport() {
    if (!state.activeResult) return;
    const r = state.activeResult;

    const report = {
        source: r.source,
        timestamp: r.timestamp,
        screening_verdict: r.prediction === 'Planet' ? 'Exoplanet Candidate' : 'Non-Candidate',
        raw_prediction: r.prediction,
        planet_probability: r.probability,
        screening_confidence_pct: r.confidence,
        primary_model: r.model,
        selection_metric: r.selection_metric,
        metrics: r.metrics || {},
        model_probabilities: r.model_probabilities || {},
        features: r.features || {}
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'exodip_report.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ==========================================
// 9. App Initialization
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    // Fix the sample CSV download link to point to the right backend
    const sampleLink = document.querySelector('a[href="/api/sample-csv"]');
    if (sampleLink) {
        sampleLink.href = BACKEND_URL + '/api/sample-csv';
    }

    // Silent wake-up ping — fires immediately so Render starts booting
    // while the user is still reading the page. No UI blocking.
    fetch(BACKEND_URL + '/api/health')
        .then(res => {
            const isOk = res.ok;
            const color = isOk ? '#34d399' : '#f43f5e';
            const shadow = isOk ? '0 0 8px rgba(52, 211, 153, 0.6)' : '0 0 8px rgba(244, 63, 94, 0.6)';
            const text = isOk ? 'AI Core: Online' : 'AI Core: Offline';

            document.querySelectorAll('.status-dot').forEach(d => {
                d.style.background = color;
                d.style.boxShadow = shadow;
            });
            const statusText = document.getElementById('api-status-text');
            if (statusText) statusText.textContent = text;
            const mobileStatusText = document.querySelector('.mobile-status-text');
            if (mobileStatusText) mobileStatusText.textContent = text;
        })
        .catch(() => {
            document.querySelectorAll('.status-dot').forEach(d => {
                d.style.background = '#f43f5e';
                d.style.boxShadow = '0 0 8px rgba(244, 63, 94, 0.6)';
            });
            const statusText = document.getElementById('api-status-text');
            if (statusText) statusText.textContent = 'AI Core: Offline';
            const mobileStatusText = document.querySelector('.mobile-status-text');
            if (mobileStatusText) mobileStatusText.textContent = 'AI Core: Offline';
        });

    // Fetch test-set metadata for the row-ID hint
    fetch(BACKEND_URL + '/api/test-set-info')
        .then(res => res.json())
        .then(info => {
            if (info && info.max_row_id != null) {
                const hint = document.getElementById('test-row-hint');
                const input = document.getElementById('test-row-id');
                if (hint) hint.textContent = `Available range: 0 to ${info.max_row_id}`;
                if (input) input.max = info.max_row_id;
            }
        })
        .catch(err => console.warn('Could not fetch test set info:', err));

    // Handle mobile orientation changes & window resize for Plotly charts
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            if (window.Plotly) {
                ['chart-full', 'chart-zoom', 'chart-probs'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el && el.data) {
                        Plotly.Plots.resize(el);
                    }
                });
            }
        }, 150);
    });
});

