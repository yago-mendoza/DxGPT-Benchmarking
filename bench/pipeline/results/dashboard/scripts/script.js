// script.js
(function() {
    'use strict';

    // Global state (within this IIFE)
    let experiments = new Map();
    let selectedExperiments = new Set();
    let comparisonChart = null;
    let detailCharts = new Map(); // Stores chart instances for detail view: Map<cellIndex, ChartInstance>
    let currentView = 'comparison';

    // DOM Elements
    const DOM = {
        sidebar: document.getElementById('sidebar'),
        sidebarToggle: document.getElementById('sidebarToggle'),
        experimentsList: document.getElementById('experimentsList'),
        comparisonView: document.getElementById('comparisonView'),
        detailView: document.getElementById('detailView'),
        comparisonChartCanvas: document.getElementById('comparisonChart'),
        gridLayoutSelector: document.getElementById('gridLayout'),
        gridContainer: document.getElementById('gridContainer'),
        viewTabs: document.querySelectorAll('.view-tab'),
        exportViewBtn: document.getElementById('exportViewBtn'),
        openJsonExplorerBtn: document.getElementById('openJsonExplorerBtn'),
        savePlotsBtn: document.getElementById('savePlotsBtn'),
        jsonModal: document.getElementById('jsonModal'),
        closeJsonExplorerBtn: document.getElementById('closeJsonExplorerBtn'),
        jsonFileSelector: document.getElementById('jsonFileSelector'),
        jsonViewer: document.getElementById('jsonViewer')
    };

    // Chart.js defaults
    Chart.defaults.color = '#e5e7eb'; // --text-primary
    Chart.defaults.borderColor = '#2a2a2a'; // --border-secondary
    Chart.defaults.backgroundColor = '#1a1a1a'; // --bg-card (for tooltips etc.)
    Chart.defaults.font.family = 'Inter, sans-serif';


    // --- INITIALIZATION ---
    async function initDashboard() {
        console.log('Initializing dashboard...');
        await loadExperiments();
        setupEventListeners();
        updateViews(); // Initial view render
    }

    // --- DATA LOADING ---
    async function loadExperiments() {
        try {
            const experimentPaths = await discoverExperiments();
            if (!experimentPaths || experimentPaths.length === 0) {
                DOM.experimentsList.innerHTML = '<div class="error">No experiment paths discovered.</div>';
                return;
            }

            const experimentPromises = experimentPaths.map(path => loadExperiment(path).catch(err => {
                console.warn(`Failed to load experiment ${path}:`, err);
                return null; // Ensure Promise.all doesn't fail on one error
            }));
            
            const loadedExperimentsArray = await Promise.all(experimentPromises);
            
            loadedExperimentsArray.forEach(experiment => {
                if (experiment) {
                    experiments.set(experiment.id, experiment);
                }
            });

            renderExperimentsList();
        } catch (err) {
            console.error('Failed to load experiments:', err);
            DOM.experimentsList.innerHTML = `<div class="error">Failed to load experiments list: ${err.message}</div>`;
        }
    }

    async function discoverExperiments() {
        const knownExperiments = [];
        // API First
        try {
            const apiResponse = await fetch('/api/experiments'); // Adjust if your API endpoint is different
            if (apiResponse.ok) {
                const experimentList = await apiResponse.json();
                console.log('Loaded experiments from API:', experimentList);
                for (const exp of experimentList) {
                    try {
                        // Assuming exp is just the folder name like "experiment_gpt-3.5-turbo_20230101120000"
                        // The dashboard is in results/dashboard/scripts/, experiments are in results/experiment_...
                        // So, paths should be relative from results/dashboard/scripts/ - need to go up 2 levels
                        const response = await fetch(`../../${exp}/summary.json`, { method: 'HEAD' });
                        if (response.ok) {
                            knownExperiments.push(exp); // Store the path relative to `results/`
                        }
                    } catch (err) {
                        console.warn(`Experiment ${exp} not accessible via API check`);
                    }
                }
                if (knownExperiments.length > 0) return knownExperiments;
            }
        } catch (err) {
            console.warn('Could not load experiments from API, trying fallback methods.', err);
        }

        // Fallback: Directory listing (requires server configuration)
        // This is highly dependent on server setup and often disabled for security.
        try {
            const response = await fetch('../../'); // Fetch parent directory (results/)
            if (response.ok) {
                const text = await response.text();
                const parser = new DOMParser();
                const doc = parser.parseFromString(text, 'text/html');
                const links = doc.querySelectorAll('a');
                links.forEach(link => {
                    const href = link.getAttribute('href');
                    // Regex for experiment_model-name_timestamp/
                    if (href && /^experiment_.*_\d{14}\/$/.test(href)) {
                        knownExperiments.push(href.slice(0, -1)); // Remove trailing slash
                    }
                });
            }
        } catch (err) {
            console.warn('Could not fetch directory listing. Ensure server allows it or use API.', err);
        }
        
        console.log('Discovered experiments:', knownExperiments);
        if (knownExperiments.length === 0) {
             console.warn("No experiments found. Ensure experiments are in sibling directories to 'dashboard' (e.g., results/experiment_xyz, results/dashboard/) and accessible.");
        }
        return knownExperiments;
    }

    async function loadExperiment(experimentFolderName) {
        // Paths are relative from the results/dashboard/scripts/ directory, where index.html is located.
        // So, experiment data is in ../../experiment_folder_name/
        const basePath = `../../${experimentFolderName}`; 
        
        try {
            const experiment = {
                id: experimentFolderName,
                name: extractModelName(experimentFolderName),
                timestamp: extractTimestamp(experimentFolderName),
                path: basePath, // Store the relative path for fetching files
                summary: null,
                semantic: null,
                severity: null
            };

            const filesToLoad = [
                { key: 'summary', path: `${basePath}/summary.json` },
                { key: 'semantic', path: `${basePath}/semantic_evaluation.json` },
                { key: 'severity', path: `${basePath}/severity_evaluation.json` }
            ];

            await Promise.all(filesToLoad.map(async fileInfo => {
                try {
                    const response = await fetch(fileInfo.path);
                    if (response.ok) {
                        experiment[fileInfo.key] = await response.json();
                    } else {
                         console.warn(`Could not load ${fileInfo.path} (Status: ${response.status}) for ${experimentFolderName}`);
                    }
                } catch (err) {
                    console.warn(`Error fetching ${fileInfo.path} for ${experimentFolderName}:`, err);
                }
            }));
            
            return experiment;
        } catch (err) {
            console.error(`Error processing experiment ${experimentFolderName}:`, err);
            return null;
        }
    }

    function extractModelName(path) {
        // experiment_model-name_timestamp
        const parts = path.replace(/^experiment_/, '').split('_');
        if (parts.length > 1) { // Has at least model and timestamp
            // Join all parts except the last one (timestamp)
            return parts.slice(0, -1).join(' ')
                        .replace(/-/g, ' ') // Replace hyphens with spaces
                        .replace(/\b\w/g, l => l.toUpperCase()); // Capitalize words
        }
        return path;
    }

    function extractTimestamp(path) {
        const match = path.match(/_(\d{14})$/);
        if (match) {
            const ts = match[1];
            return `${ts.substring(0, 4)}-${ts.substring(4, 6)}-${ts.substring(6, 8)} ${ts.substring(8, 10)}:${ts.substring(10, 12)}`;
        }
        return 'N/A';
    }

    // --- UI RENDERING ---
    function renderExperimentsList() {
        DOM.experimentsList.innerHTML = ''; // Clear previous items

        if (experiments.size === 0) {
            DOM.experimentsList.innerHTML = '<div class="loading">No experiments found or loaded.</div>';
            return;
        }

        // Sort experiments by timestamp descending (newest first)
        const sortedExperiments = Array.from(experiments.values()).sort((a, b) => {
            const tsA = a.id.match(/_(\d{14})$/)?.[1] || '0';
            const tsB = b.id.match(/_(\d{14})$/)?.[1] || '0';
            return tsB.localeCompare(tsA);
        });

        sortedExperiments.forEach(exp => {
            const semanticScore = exp.summary?.semantic_evaluation?.mean_score;
            const severityScore = exp.summary?.severity_evaluation?.mean_score;
            
            const item = document.createElement('div');
            item.className = 'experiment-item';
            item.dataset.experimentId = exp.id; // For event delegation
            item.innerHTML = `
                <div class="experiment-checkbox">
                    <input type="checkbox" id="exp-checkbox-${exp.id}" data-experiment-id="${exp.id}" />
                    <label class="experiment-info" for="exp-checkbox-${exp.id}">
                        <div class="experiment-name">${exp.name}</div>
                        <div class="experiment-timestamp">${exp.timestamp}</div>
                        <div class="experiment-scores">
                            <div class="score-item">
                                <span class="score-label">Semantic:</span>
                                <span class="score-value">${typeof semanticScore === 'number' ? semanticScore.toFixed(4) : '-'}</span>
                            </div>
                            <div class="score-item">
                                <span class="score-label">Severity:</span>
                                <span class="score-value">${typeof severityScore === 'number' ? severityScore.toFixed(4) : '-'}</span>
                            </div>
                        </div>
                    </label>
                </div>
            `;
            DOM.experimentsList.appendChild(item);
        });
    }

    function updateViews() {
        if (currentView === 'comparison') {
            DOM.comparisonView.style.display = 'flex';
            DOM.detailView.style.display = 'none';
            updateComparisonView();
        } else {
            DOM.comparisonView.style.display = 'none';
            DOM.detailView.style.display = 'block';
            updateDetailViewGrid(); // Renamed for clarity
        }
    }

    function updateComparisonView() {
        if (comparisonChart) {
            comparisonChart.destroy();
        }

        const datasets = [];
        const colors = ['#dc2626', '#ef4444', '#f87171', '#fca5a5', '#fecaca']; // Accent colors
        const markers = ['circle', 'triangle', 'rect', 'star', 'crossRot'];
        let colorIndex = 0;

        selectedExperiments.forEach(expId => {
            const exp = experiments.get(expId);
            if (exp && exp.summary) {
                const semanticScore = exp.summary.semantic_evaluation?.mean_score || 0;
                const severityScore = exp.summary.severity_evaluation?.mean_score || 0;
                
                datasets.push({
                    label: exp.name,
                    data: [{ x: severityScore, y: semanticScore }],
                    backgroundColor: colors[colorIndex % colors.length],
                    borderColor: colors[colorIndex % colors.length],
                    borderWidth: 2,
                    pointStyle: markers[colorIndex % markers.length],
                    pointRadius: 10,
                    pointHoverRadius: 12
                });
                colorIndex++;
            }
        });

        comparisonChart = new Chart(DOM.comparisonChartCanvas.getContext('2d'), {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { usePointStyle: true, padding: 20 } },
                    tooltip: {
                        callbacks: {
                            label: ctx => `${ctx.dataset.label}: Semantic=${ctx.parsed.y.toFixed(4)}, Severity=${ctx.parsed.x.toFixed(4)}`
                        }
                    },
                    zoom: {
                        pan: { enabled: true, mode: 'xy' },
                        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }
                    }
                },
                scales: {
                    x: { 
                        type: 'linear', position: 'bottom', reverse: true, min: 0, max: 1,
                        title: { display: true, text: 'Severity Score (Lower is better)', color: '#ffffff' },
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.2)' }
                    },
                    y: { 
                        type: 'linear', position: 'left', min: 0, max: 1,
                        title: { 
                            display: true, 
                            text: 'Semantic Score (Higher is better)', 
                            color: '#ffffff',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        ticks: { 
                            color: '#ffffff',
                            stepSize: 0.1,
                            min: 0,
                            max: 1,
                            precision: 1,
                            display: true,
                            callback: function(value, index, ticks) {
                                return value.toFixed(1);
                            },
                            font: {
                                size: 12
                            }
                        },
                        grid: { 
                            color: 'rgba(255, 255, 255, 0.2)',
                            display: true
                        },
                        display: true
                    } 
                }
            }
        });
    }

    function updateDetailViewGrid() {
        const layout = DOM.gridLayoutSelector.value;
        DOM.gridContainer.className = `grid-container grid-${layout}`;
        DOM.gridContainer.innerHTML = ''; // Clear existing cells
        detailCharts.forEach(chart => chart.destroy()); // Destroy old chart instances
        detailCharts.clear();

        let cellCount = 1;
        if (layout === '1x2') cellCount = 2;
        else if (layout === '2x2') cellCount = 4;

        for (let i = 0; i < cellCount; i++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            const canvasId = `detailChart${i}`;
            cell.innerHTML = `
                <div class="cell-controls">
                    <select class="experiment-selector" data-cell-index="${i}">
                        <option value="">Select experiment...</option>
                        ${Array.from(selectedExperiments).map(expId => {
                            const exp = experiments.get(expId);
                            return `<option value="${expId}">${exp.name}</option>`;
                        }).join('')}
                    </select>
                    <select class="chart-selector" data-cell-index="${i}" disabled>
                        <option value="">Select chart...</option>
                    </select>
                </div>
                <div class="chart-wrapper">
                    <canvas id="${canvasId}"></canvas>
                </div>
            `;
            DOM.gridContainer.appendChild(cell);
        }
    }

    function populateChartOptions(chartSelectorEl, expId) {
        const chartGroups = {
            'Summary': [{ value: 'stats-summary', label: 'Statistical Summary' }],
            'Combined': [{ value: 'semantic-bias', label: 'Combined Bias Evaluation' }],
            'Semantic Analysis': [
                { value: 'semantic-histogram', label: 'Score Distribution with KDE' },
                { value: 'ridge-plot', label: 'Ridge Plot by Severity' }
            ],
            'Severity Analysis': [
                { value: 'gdx-ddx-severity', label: 'GDX vs DDX Severity' },
                { value: 'severity-levels', label: 'Severity Levels Distribution' },
                { value: 'optimist-pessimist', label: 'Optimist vs Pessimist Balance' }
            ]
        };

        chartSelectorEl.innerHTML = '<option value="">Select chart...</option>';
        Object.entries(chartGroups).forEach(([groupName, charts]) => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = groupName;
            charts.forEach(chartInfo => {
                const option = document.createElement('option');
                option.value = chartInfo.value;
                option.textContent = chartInfo.label;
                optgroup.appendChild(option);
            });
            chartSelectorEl.appendChild(optgroup);
        });
    }

    // --- EVENT LISTENERS ---
    function setupEventListeners() {
        DOM.sidebarToggle.addEventListener('click', () => {
            DOM.sidebar.classList.toggle('collapsed');
            // On mobile, 'collapsed' might mean fully hidden. We need an 'open' class for mobile.
            if (window.innerWidth <= 768 && !DOM.sidebar.classList.contains('collapsed')) {
                DOM.sidebar.classList.add('open');
            } else if (window.innerWidth <= 768) {
                DOM.sidebar.classList.remove('open');
            }
            // Trigger chart resize after sidebar animation
            setTimeout(() => {
                if (comparisonChart) comparisonChart.resize();
                detailCharts.forEach(chart => chart.resize());
            }, 300); 
        });

        DOM.experimentsList.addEventListener('click', (e) => {
            const targetItem = e.target.closest('.experiment-item');
            if (!targetItem) return;

            const checkbox = targetItem.querySelector('input[type="checkbox"]');
            const experimentId = checkbox.dataset.experimentId;

            // If click was not on checkbox itself, toggle it
            if (e.target !== checkbox) {
                checkbox.checked = !checkbox.checked;
            }
            
            if (checkbox.checked) {
                selectedExperiments.add(experimentId);
                targetItem.classList.add('selected');
            } else {
                selectedExperiments.delete(experimentId);
                targetItem.classList.remove('selected');
            }
            updateViews();
             // If detail view is active, repopulate experiment selectors in grid cells
            if (currentView === 'detail') {
                updateDetailViewGrid(); 
            }
        });

        DOM.viewTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                DOM.viewTabs.forEach(t => t.classList.remove('active'));
                e.currentTarget.classList.add('active');
                currentView = e.currentTarget.dataset.view;
                updateViews();
            });
        });

        DOM.gridLayoutSelector.addEventListener('change', () => updateDetailViewGrid());

        DOM.gridContainer.addEventListener('change', (e) => {
            const target = e.target;
            const cellIndex = target.dataset.cellIndex;

            if (target.classList.contains('experiment-selector')) {
                const chartSelector = DOM.gridContainer.querySelector(`.chart-selector[data-cell-index="${cellIndex}"]`);
                if (target.value) {
                    chartSelector.disabled = false;
                    populateChartOptions(chartSelector, target.value);
                } else {
                    chartSelector.disabled = true;
                    chartSelector.innerHTML = '<option value="">Select chart...</option>';
                    if (detailCharts.has(cellIndex)) {
                        detailCharts.get(cellIndex).destroy();
                        detailCharts.delete(cellIndex);
                    }
                }
            } else if (target.classList.contains('chart-selector')) {
                const experimentSelector = DOM.gridContainer.querySelector(`.experiment-selector[data-cell-index="${cellIndex}"]`);
                if (target.value && experimentSelector.value) {
                    renderDetailChart(cellIndex, experimentSelector.value, target.value);
                } else if (detailCharts.has(cellIndex)) {
                     detailCharts.get(cellIndex).destroy();
                     detailCharts.delete(cellIndex);
                }
            }
        });
        
        DOM.exportViewBtn.addEventListener('click', exportCurrentView);
        DOM.openJsonExplorerBtn.addEventListener('click', openJsonExplorer);
        DOM.savePlotsBtn.addEventListener('click', saveToExperiment);
        DOM.closeJsonExplorerBtn.addEventListener('click', closeJsonExplorer);

        DOM.jsonFileSelector.addEventListener('change', (e) => {
            if (e.target.value) {
                const [expId, file] = e.target.value.split('|');
                loadAndDisplayJson(expId, file);
            } else {
                DOM.jsonViewer.textContent = '';
            }
        });
    }

    // --- CHART RENDERING (Detail View) ---
    // This function now acts as a router to the specific, "fixed" chart functions
    function renderDetailChart(cellIndex, expId, chartType) {
        const canvas = document.getElementById(`detailChart${cellIndex}`);
        if (!canvas) {
            console.error(`Canvas for cell ${cellIndex} not found.`);
            return;
        }
        const ctx = canvas.getContext('2d');

        if (detailCharts.has(cellIndex)) {
            detailCharts.get(cellIndex).destroy();
        }

        const exp = experiments.get(expId);
        if (!exp) {
            console.error(`Experiment ${expId} not found.`);
            const phChart = createPlaceholderChart(ctx, 'Experiment data not found.');
            detailCharts.set(cellIndex, phChart);
            return;
        }
        
        console.log(`📊 Rendering ${chartType} for ${exp.name} in cell ${cellIndex}`);

        let chartInstance;
        switch (chartType) {
            case 'stats-summary':
                chartInstance = createStatsSummaryChart(ctx, exp);
                break;
            case 'semantic-histogram':
                chartInstance = createSemanticHistogramChart(ctx, exp);
                break;
            case 'severity-levels':
                chartInstance = createSeverityLevelsChart(ctx, exp);
                break;
            case 'optimist-pessimist':
                chartInstance = createOptimistPessimistChart(ctx, exp);
                break;
            case 'semantic-bias': // This one was already using real data in your example
                chartInstance = createSemanticBiasChart(ctx, exp);
                break;
            case 'ridge-plot':
                chartInstance = createRidgePlotChart(ctx, exp);
                break;
            case 'gdx-ddx-severity':
                chartInstance = createGDXvsDDXChart(ctx, exp);
                break;
            default:
                console.warn(`Unknown chart type: ${chartType}`);
                chartInstance = createPlaceholderChart(ctx, `Unknown chart type: ${chartType}`);
        }
        detailCharts.set(cellIndex, chartInstance);
    }
    
    // --- INDIVIDUAL CHART CREATION FUNCTIONS (using "Fixed" logic) ---

    function createPlaceholderChart(ctx, message = "Chart Placeholder") {
        return new Chart(ctx, {
            type: 'bar', // or line
            data: {
                labels: ['N/A'],
                datasets: [{ label: message, data: [1] }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: {display: true, text: message, color: 'var(--text-secondary)'} },
                scales: { y: { display: false }, x: { display: false } }
            }
        });
    }

    function calculateKDE(data, bandwidth, min, max, steps = 100) {
        if (!data || data.length === 0) return [];
        const kdePoints = [];
        const stepSize = (max - min) / steps;
        for (let i = 0; i <= steps; i++) {
            const x = min + i * stepSize;
            let sum = 0;
            data.forEach(d => {
                const diff = (x - d) / bandwidth;
                sum += Math.exp(-0.5 * diff * diff) / Math.sqrt(2 * Math.PI); // Gaussian kernel
            });
            kdePoints.push({ x: x, y: sum / (data.length * bandwidth) });
        }
        return kdePoints;
    }
    
    function extractSemanticScores(exp) {
        if (!exp.semantic || !exp.semantic.evaluations) return [];
        return exp.semantic.evaluations.map(ev => ev.best_match?.score || 0).filter(s => typeof s === 'number');
    }

    function extractSeverityScores(exp) {
        if (!exp.severity || !exp.severity.evaluations) return [];
        return exp.severity.evaluations.map(ev => ev.final_score || 0).filter(s => typeof s === 'number');
    }

    function createStatsSummaryChart(ctx, exp) {
        const semanticScores = extractSemanticScores(exp);
        const severityScores = extractSeverityScores(exp);

        if (semanticScores.length === 0 && severityScores.length === 0) {
            return createPlaceholderChart(ctx, 'No summary data available.');
        }

        const calcStats = (scores) => {
            if (scores.length === 0) return { mean: 0, std: 0, count: 0 };
            const mean = ss.mean(scores);
            const std = ss.standardDeviation(scores);
            return { mean, std, count: scores.length };
        };

        const semanticStats = calcStats(semanticScores);
        const severityStats = calcStats(severityScores);
        
        // Generate vertical normal distribution curves - ONLY RIGHT SIDE
        function generateVerticalNormalCurve(mean, std, barIndex, curveWidth = 0.35) {
            const rightPoints = [];
            const steps = 80;
            
            // Ensure minimum std for visualization
            const visualStd = Math.max(std, 0.05);
            const visualRange = 3 * visualStd;
            
            for (let i = 0; i <= steps; i++) {
                // Y va de μ-3σ a μ+3σ (verticalmente)
                const y = Math.max(0, Math.min(1, mean - visualRange + (2 * visualRange * i / steps)));
                
                // Calcular densidad normal en este punto Y
                const normalizedY = (y - mean) / visualStd;
                const density = Math.exp(-0.5 * normalizedY * normalizedY);
                
                // Escalar la densidad para el ancho visual
                const scaledDensity = density * curveWidth;
                
                // Solo puntos del lado derecho
                rightPoints.push({ x: barIndex + scaledDensity, y: y });
            }
            
            return rightPoints;
        }

        const semanticCurve = generateVerticalNormalCurve(semanticStats.mean, semanticStats.std, 0, 0.5);
        const severityCurve = generateVerticalNormalCurve(severityStats.mean, severityStats.std, 1, 0.5);
        
        const data = {
            labels: ['Semantic', 'Severity'],
            datasets: [{
                label: 'Mean Score',
                data: [semanticStats.mean, severityStats.mean],
                backgroundColor: [ 'rgba(99, 102, 241, 0.7)', 'rgba(239, 68, 68, 0.7)' ],
                borderColor: [ 'rgba(99, 102, 241, 1)', 'rgba(239, 68, 68, 1)' ],
                borderWidth: 1
            }]
        };

        // Plugin para dibujar las distribuciones normales verticales - SOLO LADO DERECHO
        const normalDistributionPlugin = {
            id: 'verticalNormalDistribution',
            afterDatasetsDraw: function(chart) {
                const ctx = chart.ctx;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                
                ctx.save();
                
                // Función para dibujar una distribución (flat izquierda + curva derecha)
                const drawDistribution = (curve, stats, barIndex, color) => {
                    if (stats.std <= 0) return;
                    
                    const visualStd = Math.max(stats.std, 0.05);
                    const visualRange = 3 * visualStd;
                    const yMin = Math.max(0, Math.min(1, stats.mean - visualRange));
                    const yMax = Math.max(0, Math.min(1, stats.mean + visualRange));
                    
                    // Configurar estilo
                    ctx.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.6)`;
                    ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.12)`;
                    ctx.lineWidth = 1.8;
                    
                    // Crear path para relleno: flat izquierda + curva derecha
                    ctx.beginPath();
                    
                    // Empezar desde el punto inferior izquierdo (centro de barra)
                    const centerX = xScale.getPixelForValue(barIndex);
                    const yMinPx = yScale.getPixelForValue(yMin);
                    const yMaxPx = yScale.getPixelForValue(yMax);
                    
                    ctx.moveTo(centerX, yMinPx);
                    ctx.lineTo(centerX, yMaxPx); // Línea vertical plana izquierda
                    
                    // Conectar con la curva derecha (de arriba hacia abajo)
                    for (let i = curve.length - 1; i >= 0; i--) {
                        const point = curve[i];
                        const x = xScale.getPixelForValue(point.x);
                        const y = yScale.getPixelForValue(point.y);
                        ctx.lineTo(x, y);
                    }
                    
                    ctx.closePath();
                    ctx.fill();
                    
                    // Dibujar contornos
                    ctx.beginPath();
                    ctx.moveTo(centerX, yMinPx);
                    ctx.lineTo(centerX, yMaxPx); // Línea plana izquierda
                    ctx.stroke();
                    
                    ctx.beginPath();
                    curve.forEach((point, i) => {
                        const x = xScale.getPixelForValue(point.x);
                        const y = yScale.getPixelForValue(point.y);
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    });
                    ctx.stroke();
                    
                    // Líneas de referencia para ±1σ, ±2σ
                    ctx.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.35)`;
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 3]);
                    
                    [1, 2].forEach(sigma => {
                        const yPos1 = Math.max(0, Math.min(1, stats.mean + sigma * visualStd));
                        const yPos2 = Math.max(0, Math.min(1, stats.mean - sigma * visualStd));
                        
                        [yPos1, yPos2].forEach(yPos => {
                            if (yPos >= 0 && yPos <= 1) {
                                const yPx = yScale.getPixelForValue(yPos);
                                const xLeft = xScale.getPixelForValue(barIndex - 0.4);
                                const xRight = xScale.getPixelForValue(barIndex + 0.4);
                                
                                ctx.beginPath();
                                ctx.moveTo(xLeft, yPx);
                                ctx.lineTo(xRight, yPx);
                                ctx.stroke();
                            }
                        });
                    });
                    
                    ctx.setLineDash([]); // Reset dash
                };
                
                // Dibujar distribución para Semantic (azul)
                drawDistribution(semanticCurve, semanticStats, 0, [99, 102, 241]);
                
                // Dibujar distribución para Severity (rojo)
                drawDistribution(severityCurve, severityStats, 1, [239, 68, 68]);
                
                ctx.restore();
            }
        };

        return new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                responsive: true, 
                maintainAspectRatio: false,
                clip: false,
                layout: {
                    padding: {
                        top: 25,
                        bottom: 25,
                        left: 50,
                        right: 45
                    }
                },
                plugins: {
                    title: { 
                        display: true, 
                        text: `Statistical Summary with Right-Side Normal Distributions (Sem: ${semanticStats.count}, Sev: ${severityStats.count} evals)` 
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const stats = context.dataIndex === 0 ? semanticStats : severityStats;
                                return [
                                    `Mean: ${stats.mean.toFixed(3)}`,
                                    `StdDev: ${stats.std.toFixed(3)}`,
                                    `Range: [${Math.max(0, stats.mean - 3*stats.std).toFixed(3)}, ${Math.min(1, stats.mean + 3*stats.std).toFixed(3)}]`
                                ];
                            }
                        }
                    }
                },
                scales: { 
                    x: {
                        title: { display: true, text: 'Score Type', color: '#ffffff' },
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.2)' }
                    },
                    y: { 
                        beginAtZero: true, 
                        max: 1,
                        min: 0,
                        title: { 
                            display: true, 
                            text: 'Score Value (0.0 - 1.0)', 
                            color: '#ffffff',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        ticks: { 
                            color: '#ffffff',
                            stepSize: 0.1,
                            precision: 1,
                            callback: function(value) {
                                return value.toFixed(1);
                            },
                            font: {
                                size: 12
                            }
                        },
                        grid: { 
                            color: 'rgba(255, 255, 255, 0.2)',
                            display: true
                        }
                    } 
                }
            },
            plugins: [normalDistributionPlugin]
        });
    }

    function createSemanticHistogramChart(ctx, exp) {
        const scores = extractSemanticScores(exp);
        if (scores.length === 0) return createPlaceholderChart(ctx, 'No semantic scores for histogram.');

        const bins = ss.histogram(scores, {min:0, max:1, binSize: 0.04}); // simple-statistics for binning
        const labels = bins.map(bin => `${bin.x0.toFixed(2)}-${bin.x1.toFixed(2)}`);
        const data = bins.map(bin => bin.length);

        const kdeValues = calculateKDE(scores, 0.05, 0, 1, labels.length);
        const maxHistCount = Math.max(...data, 0);
        const maxKDEVal = Math.max(...kdeValues.map(d => d.y), 0);
        
        const kdeScaled = kdeValues.map(point => ({
            x: labels[Math.floor(point.x / 0.04)] || labels[labels.length-1], // Align with bin labels
            y: maxKDEVal > 0 ? (point.y / maxKDEVal) * maxHistCount * 1.1 : 0
        }));

        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: `Score Count (n=${scores.length})`,
                        data: data,
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        order: 2
                    },
                    {
                        label: 'KDE',
                        data: kdeScaled,
                        type: 'line',
                        borderColor: 'rgba(245, 158, 11, 0.9)',
                        tension: 0.4,
                        fill: false,
                        pointRadius: 0,
                        order: 1
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { title: { display: true, text: `Semantic Score Distribution (μ=${ss.mean(scores).toFixed(3)})` } },
                scales: {
                    x: { title: { display: true, text: 'Semantic Score Range' } },
                    y: { title: { display: true, text: 'Count / Density' } }
                }
            }
        });
    }

    function createSemanticBiasChart(ctx, exp) {
        const scatterData = [];
        let maxScore = 0;

        if (exp.semantic && exp.semantic.evaluations && exp.severity && exp.severity.evaluations) {
            const semanticEvals = exp.semantic.evaluations;
            const severityMap = new Map(exp.severity.evaluations.map(s => [s.id, s]));

            semanticEvals.forEach((semEval, idx) => {
                const caseId = semEval.case_id || (idx + 1).toString();
                const severityEval = severityMap.get(caseId);

                if (severityEval) {
                    const semanticScore = semEval.best_match?.score || 0;
                    const finalScore = severityEval.final_score || 0;
                    maxScore = Math.max(maxScore, Math.abs(finalScore)); // Use abs for symmetric scaling

                    const optimistN = severityEval.optimist?.n || 0;
                    const pessimistN = severityEval.pessimist?.n || 0;
                    
                    let type = 'neutral';
                    let xValue = 0;
                    if (optimistN > pessimistN) {
                        type = 'optimist';
                        xValue = -finalScore; // Optimist on left
                    } else if (pessimistN > optimistN) {
                        type = 'pessimist';
                        xValue = finalScore; // Pessimist on right
                    }
                    scatterData.push({ x: xValue, y: semanticScore, type: type, caseId: caseId });
                }
            });
        } else {
            return createPlaceholderChart(ctx, 'Missing data for Semantic Bias Chart.');
        }

        if (scatterData.length === 0) return createPlaceholderChart(ctx, 'No data for Semantic Bias Chart.');
        maxScore = maxScore === 0 ? 0.5 : maxScore; // Ensure non-zero max for scaling

        // Calculate means for each group
        const globalMean = { x: 0, y: 0, count: 0 };
        const optimistMean = { x: 0, y: 0, count: 0 };
        const pessimistMean = { x: 0, y: 0, count: 0 };
        const neutralMean = { x: 0, y: 0, count: 0 };

        scatterData.forEach(point => {
            globalMean.x += point.x;
            globalMean.y += point.y;
            globalMean.count++;

            if (point.type === 'optimist') {
                optimistMean.x += point.x;
                optimistMean.y += point.y;
                optimistMean.count++;
            } else if (point.type === 'pessimist') {
                pessimistMean.x += point.x;
                pessimistMean.y += point.y;
                pessimistMean.count++;
            } else {
                neutralMean.x += point.x;
                neutralMean.y += point.y;
                neutralMean.count++;
            }
        });

        // Calculate final means
        if (globalMean.count > 0) {
            globalMean.x /= globalMean.count;
            globalMean.y /= globalMean.count;
        }
        if (optimistMean.count > 0) {
            optimistMean.x /= optimistMean.count;
            optimistMean.y /= optimistMean.count;
        }
        if (pessimistMean.count > 0) {
            pessimistMean.x /= pessimistMean.count;
            pessimistMean.y /= pessimistMean.count;
        }
        if (neutralMean.count > 0) {
            neutralMean.x /= neutralMean.count;
            neutralMean.y /= neutralMean.count;
        }

        const datasets = [
            // Puntos scatter con más transparencia (en el fondo)
            { 
                label: 'Optimist Majority', 
                data: scatterData.filter(d => d.type === 'optimist'), 
                backgroundColor: 'rgba(59, 130, 246, 0.35)', // Más transparente: 0.7 → 0.35
                pointRadius: 3, // Más pequeños: 4 → 3
                order: 10 // Al fondo
            },
            { 
                label: 'Pessimist Majority', 
                data: scatterData.filter(d => d.type === 'pessimist'), 
                backgroundColor: 'rgba(239, 68, 68, 0.35)', // Más transparente: 0.7 → 0.35
                pointRadius: 3, // Más pequeños: 4 → 3
                order: 10 // Al fondo
            },
            { 
                label: 'Balanced', 
                data: scatterData.filter(d => d.type === 'neutral'), 
                backgroundColor: 'rgba(128, 128, 128, 0.35)', // Más transparente: 0.7 → 0.35
                pointRadius: 3, // Más pequeños: 4 → 3
                order: 10 // Al fondo
            }
        ];

        // Add mean points datasets (por delante y más prominentes)
        if (globalMean.count > 0) {
            datasets.push({
                label: 'Global Mean',
                data: [{ x: globalMean.x, y: globalMean.y }],
                backgroundColor: 'rgba(255, 255, 255, 0.95)', // Más opaco
                borderColor: 'rgba(0, 0, 0, 0.9)', // Más opaco
                borderWidth: 3, // Más grueso: 2 → 3
                pointStyle: 'star',
                pointRadius: 14, // Más grande: 12 → 14
                pointHoverRadius: 16,
                order: 1 // Por delante
            });
        }
        if (optimistMean.count > 0) {
            datasets.push({
                label: 'Optimist Mean',
                data: [{ x: optimistMean.x, y: optimistMean.y }],
                backgroundColor: 'rgba(59, 130, 246, 0.95)', // Más opaco
                borderColor: 'rgba(29, 78, 216, 1)',
                borderWidth: 3, // Más grueso: 2 → 3
                pointStyle: 'triangle',
                pointRadius: 12, // Más grande: 10 → 12
                pointHoverRadius: 14,
                order: 1 // Por delante
            });
        }
        if (pessimistMean.count > 0) {
            datasets.push({
                label: 'Pessimist Mean',
                data: [{ x: pessimistMean.x, y: pessimistMean.y }],
                backgroundColor: 'rgba(239, 68, 68, 0.95)', // Más opaco
                borderColor: 'rgba(185, 28, 28, 1)',
                borderWidth: 3, // Más grueso: 2 → 3
                pointStyle: 'triangle',
                pointRadius: 12, // Más grande: 10 → 12
                pointHoverRadius: 14,
                order: 1 // Por delante
            });
        }
        if (neutralMean.count > 0) {
            datasets.push({
                label: 'Neutral Mean',
                data: [{ x: neutralMean.x, y: neutralMean.y }],
                backgroundColor: 'rgba(128, 128, 128, 0.95)', // Más opaco
                borderColor: 'rgba(64, 64, 64, 1)',
                borderWidth: 3, // Más grueso: 2 → 3
                pointStyle: 'rect',
                pointRadius: 12, // Más grande: 10 → 12
                pointHoverRadius: 14,
                order: 1 // Por delante
            });
        }

        return new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true, 
                maintainAspectRatio: false,
                plugins: { 
                    title: { 
                        display: true, 
                        text: 'Combined Bias Evaluation - Means Highlighted',
                        color: '#ffffff'
                    },
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        title: { 
                            display: true, 
                            text: '← Optimist | Severity Score Bias | Pessimist →',
                            color: '#ffffff'
                        },
                        min: -maxScore * 1.2, 
                        max: maxScore * 1.2,
                        ticks: { 
                            callback: value => Math.abs(value).toFixed(2),
                            color: '#ffffff'
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.2)' }
                    },
                    y: { 
                        title: { 
                            display: true, 
                            text: 'Semantic Level',
                            color: '#ffffff'
                        }, 
                        min: 0, 
                        max: 1,
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.2)' }
                    }
                }
            }
        });
    }

    function createRidgePlotChart(ctx, exp) {
        if (!exp.semantic || !exp.semantic.evaluations || !exp.severity || !exp.severity.evaluations) {
            return createPlaceholderChart(ctx, 'Missing data for Ridge Plot.');
        }

        const scoresByGDXSeverity = {};
        for (let i = 3; i <= 10; i++) scoresByGDXSeverity[`S${i}`] = [];

        const severityMap = new Map(exp.severity.evaluations.map(s => [s.id, s]));
        exp.semantic.evaluations.forEach((semEval, idx) => {
            const caseId = semEval.case_id || (idx + 1).toString();
            const severityEval = severityMap.get(caseId);
            if (severityEval && severityEval.gdx) {
                const gdxSeverity = severityEval.gdx.severity; // e.g., "S5"
                const semanticScore = semEval.best_match?.score || 0;
                if (scoresByGDXSeverity[gdxSeverity]) {
                    scoresByGDXSeverity[gdxSeverity].push(semanticScore);
                }
            }
        });

        const ridgePlotData = [];
        const severityLevels = Object.keys(scoresByGDXSeverity).sort((a,b) => parseInt(b.slice(1)) - parseInt(a.slice(1))); // S10, S9,...

        severityLevels.forEach((severityKey, idx) => {
            const scores = scoresByGDXSeverity[severityKey];
            if (scores.length > 5) { // Need enough points for KDE
                const kde = calculateKDE(scores, 0.05, 0, 1, 50);
                const maxDensity = Math.max(...kde.map(d => d.y), 0);
                if (maxDensity > 0) {
                    const normalizedKDE = kde.map(point => ({
                        x: point.x,
                        y: (point.y / maxDensity) * 0.8 + idx // Offset by index for ridge effect
                    }));
                    ridgePlotData.push({
                        label: `${severityKey} (n=${scores.length})`,
                        data: normalizedKDE,
                        borderColor: `hsl(${260 - idx * 25}, 70%, 60%)`, // Color gradient
                        backgroundColor: `hsla(${260 - idx * 25}, 70%, 60%, 0.3)`,
                        fill: {target: {value: idx}, above: `hsla(${260 - idx * 25}, 70%, 60%, 0.3)`},
                        tension: 0.4,
                        pointRadius: 0,
                        borderWidth: 1.5
                    });
                }
            }
        });
        
        if (ridgePlotData.length === 0) return createPlaceholderChart(ctx, 'Not enough data for Ridge Plot.');

        return new Chart(ctx, {
            type: 'line',
            data: { datasets: ridgePlotData },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { title: { display: true, text: 'Semantic Score Ridge Plot by GDX Severity' }, legend: { position: 'right' } },
                scales: {
                    x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Semantic Score' } },
                    y: { type: 'linear', min: -0.5, max: severityLevels.length -0.5, display: false } // Hide Y-axis labels
                }
            }
        });
    }

    function createGDXvsDDXChart(ctx, exp) {
        if (!exp.severity || !exp.severity.evaluations) {
            return createPlaceholderChart(ctx, 'No severity data for GDX vs DDX.');
        }

        const optimistData = [];
        const pessimistData = [];
        const perfectMatch = [];
        
        // Group data by GDX severity for boxplots
        const gdxGroups = {};
        
        exp.severity.evaluations.forEach(evaluation => {
            if (!evaluation.gdx || !evaluation.gdx.severity || !evaluation.ddx_list) return;
            
            const gdxSeverity = parseInt(evaluation.gdx.severity.replace('S', ''));
            
            if (!gdxGroups[gdxSeverity]) {
                gdxGroups[gdxSeverity] = {
                    optimist: [],
                    pessimist: [],
                    all: []
                };
            }
            
            // Calculate mean DDX severity for optimists and pessimists
            let optimistSevs = [];
            let pessimistSevs = [];
            
            evaluation.ddx_list.forEach(ddx => {
                if (!ddx.severity) return;
                const ddxSeverity = parseInt(ddx.severity.replace('S', ''));
                gdxGroups[gdxSeverity].all.push(ddxSeverity);
                
                // Pessimists: DDX severity HIGHER than GDX (above the line)
                if (ddxSeverity > gdxSeverity) {
                    pessimistSevs.push(ddxSeverity);
                    gdxGroups[gdxSeverity].pessimist.push(ddxSeverity);
                } 
                // Optimists: DDX severity LOWER than GDX (below the line)
                else if (ddxSeverity < gdxSeverity) {
                    optimistSevs.push(ddxSeverity);
                    gdxGroups[gdxSeverity].optimist.push(ddxSeverity);
                }
            });
            
            // Add mean points for each case
            if (optimistSevs.length > 0) {
                const meanOptimist = optimistSevs.reduce((a, b) => a + b, 0) / optimistSevs.length;
                optimistData.push({x: gdxSeverity, y: meanOptimist});
            }
            
            if (pessimistSevs.length > 0) {
                const meanPessimist = pessimistSevs.reduce((a, b) => a + b, 0) / pessimistSevs.length;
                pessimistData.push({x: gdxSeverity, y: meanPessimist});
            }
        });
        
        // Create perfect match line
        for (let i = 0; i <= 10; i++) {
            perfectMatch.push({x: i, y: i});
        }
        
        // Create bottom axis histogram for GDX severity distribution
        const gdxSeverityHist = {};
        for (let i = 0; i <= 10; i++) {
            gdxSeverityHist[i] = 0;
        }
        
        exp.severity.evaluations.forEach(evaluation => {
            if (evaluation.gdx && evaluation.gdx.severity) {
                const gdxSeverity = parseInt(evaluation.gdx.severity.replace('S', ''));
                gdxSeverityHist[gdxSeverity]++;
            }
        });
        
        // Create histogram bars as points at bottom
        const bottomHistogramData = [];
        const maxGdxCount = Math.max(...Object.values(gdxSeverityHist));
        const histScale = 0.6; // Height scale for histogram
        
        for (let severity = 0; severity <= 10; severity++) {
            const count = gdxSeverityHist[severity];
            if (count > 0) {
                const height = (count / maxGdxCount) * histScale;
                const barWidth = 0.6;
                const pointsPerBar = 15;
                
                for (let px = 0; px < pointsPerBar; px++) {
                    const xOffset = (px / pointsPerBar - 0.5) * barWidth;
                    for (let py = 0; py <= 8; py++) {
                        bottomHistogramData.push({
                            x: severity + xOffset,
                            y: -0.3 + (height * py / 8)
                        });
                    }
                }
            }
        }
        
        // Create simple boxplot data
        const boxplotData = [];
        const boxplotFillData = [];
        Object.keys(gdxGroups).forEach(gdxSev => {
            const gdxSeverity = parseInt(gdxSev);
            const optimistValues = gdxGroups[gdxSeverity].optimist;
            const pessimistValues = gdxGroups[gdxSeverity].pessimist;
            
            // Add minimalistic optimist boxplots (blue) - left side, more separated
            if (optimistValues.length > 0) {
                const sorted = optimistValues.sort((a, b) => a - b);
                const q1 = sorted[Math.floor(sorted.length * 0.25)];
                const median = sorted[Math.floor(sorted.length * 0.5)];
                const q3 = sorted[Math.floor(sorted.length * 0.75)];
                const min = sorted[0];
                const max = sorted[sorted.length - 1];
                
                const boxLeft = gdxSeverity - 0.35;
                const boxRight = gdxSeverity - 0.15;
                const centerX = (boxLeft + boxRight) / 2;
                
                // Store box fill data for aesthetics
                boxplotFillData.push({
                    type: 'optimist',
                    boxLeft: boxLeft,
                    boxRight: boxRight,
                    q1: q1,
                    q3: q3,
                    median: median
                });
                
                // Box outline as line segments
                boxplotData.push({x: boxLeft, y: q1, x2: boxRight, y2: q1, type: 'optimist', lineType: 'horizontal'});
                boxplotData.push({x: boxLeft, y: q3, x2: boxRight, y2: q3, type: 'optimist', lineType: 'horizontal'});
                boxplotData.push({x: boxLeft, y: q1, x2: boxLeft, y2: q3, type: 'optimist', lineType: 'vertical'});
                boxplotData.push({x: boxRight, y: q1, x2: boxRight, y2: q3, type: 'optimist', lineType: 'vertical'});
                
                // Median line
                boxplotData.push({x: boxLeft, y: median, x2: boxRight, y2: median, type: 'optimist', lineType: 'median'});
                
                // Whiskers
                boxplotData.push({x: centerX, y: min, x2: centerX, y2: q1, type: 'optimist', lineType: 'whisker'});
                boxplotData.push({x: centerX, y: q3, x2: centerX, y2: max, type: 'optimist', lineType: 'whisker'});
            }
            
            // Add minimalistic pessimist boxplots (red) - right side, more separated
            if (pessimistValues.length > 0) {
                const sorted = pessimistValues.sort((a, b) => a - b);
                const q1 = sorted[Math.floor(sorted.length * 0.25)];
                const median = sorted[Math.floor(sorted.length * 0.5)];
                const q3 = sorted[Math.floor(sorted.length * 0.75)];
                const min = sorted[0];
                const max = sorted[sorted.length - 1];
                
                const boxLeft = gdxSeverity + 0.15;
                const boxRight = gdxSeverity + 0.35;
                const centerX = (boxLeft + boxRight) / 2;
                
                // Store box fill data for aesthetics
                boxplotFillData.push({
                    type: 'pessimist',
                    boxLeft: boxLeft,
                    boxRight: boxRight,
                    q1: q1,
                    q3: q3,
                    median: median
                });
                
                // Box outline as line segments
                boxplotData.push({x: boxLeft, y: q1, x2: boxRight, y2: q1, type: 'pessimist', lineType: 'horizontal'});
                boxplotData.push({x: boxLeft, y: q3, x2: boxRight, y2: q3, type: 'pessimist', lineType: 'horizontal'});
                boxplotData.push({x: boxLeft, y: q1, x2: boxLeft, y2: q3, type: 'pessimist', lineType: 'vertical'});
                boxplotData.push({x: boxRight, y: q1, x2: boxRight, y2: q3, type: 'pessimist', lineType: 'vertical'});
                
                // Median line
                boxplotData.push({x: boxLeft, y: median, x2: boxRight, y2: median, type: 'pessimist', lineType: 'median'});
                
                // Whiskers
                boxplotData.push({x: centerX, y: min, x2: centerX, y2: q1, type: 'pessimist', lineType: 'whisker'});
                boxplotData.push({x: centerX, y: q3, x2: centerX, y2: max, type: 'pessimist', lineType: 'whisker'});
            }
        });

        // Custom plugin for drawing minimalistic boxplots with faded fills
        const boxplotPlugin = {
            id: 'minimalBoxplots',
            afterDatasetsDraw: function(chart) {
                const ctx = chart.ctx;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                
                // First draw the filled areas
                boxplotFillData.forEach(box => {
                    const leftPx = xScale.getPixelForValue(box.boxLeft);
                    const rightPx = xScale.getPixelForValue(box.boxRight);
                    const q1Px = yScale.getPixelForValue(box.q1);
                    const q3Px = yScale.getPixelForValue(box.q3);
                    const medianPx = yScale.getPixelForValue(box.median);
                    
                    ctx.save();
                    
                    const isOptimist = box.type === 'optimist';
                    const baseColor = isOptimist ? [59, 130, 246] : [239, 68, 68];
                    
                    // Upper box fill (Q3 to median) - more transparent
                    ctx.fillStyle = `rgba(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]}, 0.12)`;
                    ctx.fillRect(leftPx, q3Px, rightPx - leftPx, medianPx - q3Px);
                    
                    // Lower box fill (median to Q1) - slightly more opaque
                    ctx.fillStyle = `rgba(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]}, 0.18)`;
                    ctx.fillRect(leftPx, medianPx, rightPx - leftPx, q1Px - medianPx);
                    
                    ctx.restore();
                });
                
                // Then draw the minimal line elements
                const groupedLines = {};
                boxplotData.forEach(line => {
                    if (!line.x2) return;
                    const key = `${Math.round(line.x * 10)}:${line.type}`;
                    if (!groupedLines[key]) {
                        groupedLines[key] = {
                            type: line.type,
                            lines: []
                        };
                    }
                    groupedLines[key].lines.push(line);
                });
                
                // Draw minimal line elements for each boxplot
                Object.values(groupedLines).forEach(group => {
                    const lines = group.lines;
                    const isOptimist = group.type === 'optimist';
                    const baseColor = isOptimist ? [59, 130, 246] : [239, 68, 68];
                    
                    ctx.save();
                    
                    lines.forEach(line => {
                        const x1Px = xScale.getPixelForValue(line.x);
                        const y1Px = yScale.getPixelForValue(line.y);
                        const x2Px = xScale.getPixelForValue(line.x2);
                        const y2Px = yScale.getPixelForValue(line.y2);
                        
                        ctx.beginPath();
                        ctx.moveTo(x1Px, y1Px);
                        ctx.lineTo(x2Px, y2Px);
                        
                        // Style based on line type
                        if (line.lineType === 'median') {
                            ctx.strokeStyle = `rgba(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]}, 0.9)`;
                            ctx.lineWidth = 2;
                        } else if (line.lineType === 'whisker') {
                            ctx.strokeStyle = `rgba(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]}, 0.5)`;
                            ctx.lineWidth = 1;
                        } else {
                            ctx.strokeStyle = `rgba(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]}, 0.7)`;
                            ctx.lineWidth = 1;
                        }
                        
                        ctx.stroke();
                    });
                    
                    ctx.restore();
                });
            }
        };

        if (optimistData.length === 0 && pessimistData.length === 0) {
            return createPlaceholderChart(ctx, 'No GDX/DDX pairs found.');
        }

        return new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'GDX Severity Distribution',
                    data: bottomHistogramData,
                    backgroundColor: 'rgba(156, 163, 175, 0.25)',
                    borderColor: 'transparent',
                    pointRadius: 1.5,
                    pointHoverRadius: 1.5,
                    showLine: false,
                    order: 10
                }, {
                    label: 'Perfect Match',
                    data: perfectMatch,
                    type: 'line',
                    borderColor: 'rgba(156, 163, 175, 0.8)',
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    borderWidth: 2,
                    order: 0
                }, {
                    label: 'Mean Optimist DDX (Below Line)',
                    data: optimistData,
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    order: 1
                }, {
                    label: 'Mean Pessimist DDX (Above Line)',
                    data: pessimistData,
                    backgroundColor: 'rgba(239, 68, 68, 0.6)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    order: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                clip: false,
                layout: {
                    padding: {
                        top: 20,
                        bottom: 20,
                        left: 20,
                        right: 20
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'GDX vs DDX Severity Comparison'
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            color: 'var(--text-primary)',
                            font: {
                                size: 12
                            },
                            padding: 25,
                            boxWidth: 20,
                            boxHeight: 20,
                            usePointStyle: true
                        },
                        align: 'center',
                        fullSize: true,
                        maxHeight: 100
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'GDX Severity',
                            color: 'var(--text-primary)'
                        },
                        ticks: {
                            color: 'var(--text-secondary)',
                            stepSize: 1
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.3)'
                        },
                        min: 0,
                        max: 10
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Mean DDX Severity',
                            color: 'var(--text-primary)'
                        },
                        ticks: {
                            color: 'var(--text-secondary)',
                            stepSize: 1
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.3)'
                        },
                        min: -0.4,
                        max: 10
                    }
                }
            },
            plugins: [boxplotPlugin]
        });
    }

    function createSeverityLevelsChart(ctx, exp) {
        if (!exp.severity || !exp.severity.evaluations) {
            return createPlaceholderChart(ctx, 'No severity data for levels chart.');
        }

        const gdxCounts = Array(11).fill(0); // S0 to S10
        const ddxCounts = Array(11).fill(0);
        let totalGdx = 0;
        let totalDdx = 0;

        exp.severity.evaluations.forEach(evaluation => {
            if (evaluation.gdx && evaluation.gdx.severity) {
                const gdxSev = parseInt(evaluation.gdx.severity.replace('S', ''));
                if (gdxSev >=0 && gdxSev <=10) {
                     gdxCounts[gdxSev]++;
                     totalGdx++;
                }
            }
            if (evaluation.ddx_list) {
                evaluation.ddx_list.forEach(ddx => {
                    if (ddx.severity) {
                        const ddxSev = parseInt(ddx.severity.replace('S', ''));
                        if (ddxSev >=0 && ddxSev <=10) {
                            ddxCounts[ddxSev]++;
                            totalDdx++;
                        }
                    }
                });
            }
        });
        
        // Filter out S0-S2 as they are usually not relevant for clinical severity
        const labels = Array.from({length: 8}, (_, i) => `S${i+3}`); // S3 to S10
        const gdxData = gdxCounts.slice(3);
        const ddxData = ddxCounts.slice(3);


        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { label: `GDX (n=${totalGdx})`, data: gdxData, backgroundColor: 'rgba(16, 185, 129, 0.7)' },
                    { label: `DDX (n=${totalDdx})`, data: ddxData, backgroundColor: 'rgba(249, 115, 22, 0.7)' }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { title: { display: true, text: `Severity Level Distribution (S3-S10)` } },
                scales: {
                    x: { title: { display: true, text: 'Severity Level' } },
                    y: { title: { display: true, text: 'Count' }, beginAtZero: true }
                }
            }
        });
    }

    function createOptimistPessimistChart(ctx, exp) {
        if (!exp.severity || !exp.severity.evaluations) {
            return createPlaceholderChart(ctx, 'No severity data for optimist/pessimist.');
        }
        
        let optimistCases = 0, pessimistCases = 0, neutralCases = 0;
        const optimistByN = {1:0, 2:0, 3:0, 4:0, 5:0}; // Max n=5 for doughnut segments
        const pessimistByN = {1:0, 2:0, 3:0, 4:0, 5:0};

        exp.severity.evaluations.forEach(evaluation => {
            const optN = evaluation.optimist?.n || 0;
            const pesN = evaluation.pessimist?.n || 0;
            if (optN > pesN) {
                optimistCases++;
                optimistByN[Math.min(5, optN)]++;
            } else if (pesN > optN) {
                pessimistCases++;
                pessimistByN[Math.min(5, pesN)]++;
            } else {
                neutralCases++;
            }
        });

        const data = [
            optimistByN[1], optimistByN[2], optimistByN[3], optimistByN[4], optimistByN[5],
            pessimistByN[1], pessimistByN[2], pessimistByN[3], pessimistByN[4], pessimistByN[5]
        ];
        const labels = [
            'Opt n=1', 'Opt n=2', 'Opt n=3', 'Opt n=4', 'Opt n=5+',
            'Pes n=1', 'Pes n=2', 'Pes n=3', 'Pes n=4', 'Pes n=5+'
        ];
        const backgroundColors = [
            'rgba(59, 130, 246, 0.9)','rgba(59, 130, 246, 0.75)','rgba(59, 130, 246, 0.6)','rgba(59, 130, 246, 0.45)','rgba(59, 130, 246, 0.3)',
            'rgba(239, 68, 68, 0.9)','rgba(239, 68, 68, 0.75)','rgba(239, 68, 68, 0.6)','rgba(239, 68, 68, 0.45)','rgba(239, 68, 68, 0.3)'
        ];
        
        const centerTextPlugin = {
            id: 'centerTextPlugin',
            beforeDraw: chart => {
                const {ctx, chartArea: {top, right, bottom, left, width, height}} = chart;
                ctx.save();
                ctx.font = 'bold 1.5em Inter'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillStyle = 'var(--text-primary)';
                ctx.fillText(`${optimistCases} | ${pessimistCases}`, width / 2 + left, height / 2 + top -10);
                ctx.font = '0.9em Inter'; ctx.fillStyle = 'var(--text-secondary)';
                ctx.fillText('Optimist | Pessimist', width / 2 + left, height / 2 + top + 15);
                 if (neutralCases > 0) {
                    ctx.font = '0.7em Inter';
                    ctx.fillText(`(${neutralCases} Neutral)`, width / 2 + left, height / 2 + top + 35);
                }
                ctx.restore();
            }
        };

        return new Chart(ctx, {
            type: 'doughnut',
            data: { labels, datasets: [{ data, backgroundColor: backgroundColors, borderWidth: 1, borderColor: 'var(--bg-secondary)' }] },
            options: {
                responsive: true, maintainAspectRatio: false, cutout: '60%',
                plugins: {
                    title: { display: true, text: `Optimist vs Pessimist Balance (by N)` },
                    legend: { position: 'bottom', labels:{boxWidth:15, padding:10} },
                    tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.parsed} cases` } }
                }
            },
            plugins: [centerTextPlugin]
        });
    }

    // --- UTILITY & ACTION FUNCTIONS ---
    function exportCurrentView() {
        let canvasToExport;
        let filename = 'dashboard_view';

        if (currentView === 'comparison' && comparisonChart) {
            canvasToExport = DOM.comparisonChartCanvas;
            filename = 'comparison_view';
        } else if (currentView === 'detail') {
            const visibleCanvases = Array.from(detailCharts.values()).map(chart => chart.canvas).filter(Boolean);
            if (visibleCanvases.length === 0) {
                alert('No charts to export in detail view.');
                return;
            }
            if (visibleCanvases.length === 1) {
                canvasToExport = visibleCanvases[0];
                filename = `detail_chart_${visibleCanvases[0].id}`;
            } else {
                // Combine multiple canvases if layout is > 1x1 (simple horizontal/vertical stacking for now)
                // This part can be complex to make perfect. For now, a basic combined export.
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                const firstCanvas = visibleCanvases[0];
                const layout = DOM.gridLayoutSelector.value;

                if (layout === '1x2' && visibleCanvases.length >= 2) {
                    tempCanvas.width = firstCanvas.width * 2;
                    tempCanvas.height = firstCanvas.height;
                    tempCtx.drawImage(visibleCanvases[0], 0, 0);
                    if (visibleCanvases[1]) tempCtx.drawImage(visibleCanvases[1], firstCanvas.width, 0);
                } else if (layout === '2x2' && visibleCanvases.length >= 4) {
                    tempCanvas.width = firstCanvas.width * 2;
                    tempCanvas.height = firstCanvas.height * 2;
                    tempCtx.drawImage(visibleCanvases[0], 0, 0);
                    if (visibleCanvases[1]) tempCtx.drawImage(visibleCanvases[1], firstCanvas.width, 0);
                    if (visibleCanvases[2]) tempCtx.drawImage(visibleCanvases[2], 0, firstCanvas.height);
                    if (visibleCanvases[3]) tempCtx.drawImage(visibleCanvases[3], firstCanvas.width, firstCanvas.height);
                } else { // Fallback to first canvas if layout doesn't match available canvases
                    canvasToExport = firstCanvas;
                }
                if (!canvasToExport) canvasToExport = tempCanvas; // if tempCanvas was used
                filename = 'detail_grid_view';
            }
        }

        if (canvasToExport) {
            exportCanvas(canvasToExport, filename);
        } else {
            alert('No active chart to export.');
        }
    }

    function exportCanvas(canvas, filename) {
        canvas.toBlob(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${filename}_${new Date().toISOString().slice(0, 10).replace(/-/g,'')}.png`;
            document.body.appendChild(a); // Required for Firefox
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 'image/png');
    }

    function saveToExperiment() {
        if (selectedExperiments.size === 0) {
            alert('Please select an experiment first.');
            return;
        }
        const firstSelectedExpId = Array.from(selectedExperiments)[0];
        const exp = experiments.get(firstSelectedExpId);
        // This is a client-side app, actual saving to server requires backend.
        // For now, it could trigger downloads of all currently displayed charts.
        alert(`Simulating save plots for: ${exp.name}.\nIn a real app, this would require server-side logic or trigger individual downloads.`);
        
        // Example: download all visible detail charts if any
        if (currentView === 'detail' && detailCharts.size > 0) {
            detailCharts.forEach((chart, cellIndex) => {
                const chartType = DOM.gridContainer.querySelector(`.chart-selector[data-cell-index="${cellIndex}"]`).value;
                if (chartType) {
                    exportCanvas(chart.canvas, `${exp.id}_${chartType}_cell${cellIndex}`);
                }
            });
        } else if (currentView === 'comparison' && comparisonChart) {
             exportCanvas(comparisonChart.canvas, `${exp.id}_comparison_summary`);
        }
    }

    function openJsonExplorer() {
        DOM.jsonFileSelector.innerHTML = '<option value="">Select experiment and file...</option>';
        experiments.forEach((exp, expId) => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = exp.name;
            ['summary.json', 'semantic_evaluation.json', 'severity_evaluation.json'].forEach(file => {
                // Check if data actually exists for this file before adding option
                let dataExists = false;
                if (file === 'summary.json' && exp.summary) dataExists = true;
                if (file === 'semantic_evaluation.json' && exp.semantic) dataExists = true;
                if (file === 'severity_evaluation.json' && exp.severity) dataExists = true;
                
                if (dataExists) { // Only add if data is loaded or potentially loadable
                    const option = document.createElement('option');
                    option.value = `${expId}|${file}`;
                    option.textContent = file;
                    optgroup.appendChild(option);
                }
            });
            if (optgroup.childElementCount > 0) {
                DOM.jsonFileSelector.appendChild(optgroup);
            }
        });
        DOM.jsonViewer.textContent = 'Select an experiment and file to view its JSON content.';
        DOM.jsonModal.style.display = 'flex';
    }

    async function loadAndDisplayJson(expId, filename) {
        DOM.jsonViewer.textContent = 'Loading JSON...';
        const exp = experiments.get(expId);
        if (!exp) {
            DOM.jsonViewer.textContent = 'Error: Experiment not found.';
            return;
        }

        let jsonData = null;
        if (filename === 'summary.json') jsonData = exp.summary;
        else if (filename === 'semantic_evaluation.json') jsonData = exp.semantic;
        else if (filename === 'severity_evaluation.json') jsonData = exp.severity;

        if (jsonData) {
            DOM.jsonViewer.textContent = JSON.stringify(jsonData, null, 2);
        } else {
            // Fallback: try to fetch if not pre-loaded (should be pre-loaded by loadExperiment)
            try {
                const response = await fetch(`${exp.path}/${filename}`);
                if (response.ok) {
                    const data = await response.json();
                    DOM.jsonViewer.textContent = JSON.stringify(data, null, 2);
                    // Optionally store it back to the experiment object
                    if (filename === 'summary.json') exp.summary = data;
                    else if (filename === 'semantic_evaluation.json') exp.semantic = data;
                    else if (filename === 'severity_evaluation.json') exp.severity = data;
                } else {
                    DOM.jsonViewer.textContent = `Error: Could not load ${filename} (Status: ${response.status}). File might be missing or not accessible.`;
                }
            } catch (err) {
                DOM.jsonViewer.textContent = `Error fetching ${filename}: ${err.message}`;
            }
        }
    }

    function closeJsonExplorer() {
        DOM.jsonModal.style.display = 'none';
    }

    // --- STARTUP ---
    document.addEventListener('DOMContentLoaded', initDashboard);

})();