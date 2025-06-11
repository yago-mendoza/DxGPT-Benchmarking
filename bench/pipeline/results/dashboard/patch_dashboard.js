// Patch to fix dashboard data loading
// This script replaces the problematic chart functions with ones that use real data

// Apply this patch by including it after the main dashboard script loads

(function() {
    'use strict';
    
    console.log('🔧 Applying dashboard data fix patch...');
    
    // Override the renderDetailChart function to use fixed versions
    window.renderDetailChart = function(cellIndex, expId, chartType) {
        const canvas = document.getElementById(`detailChart${cellIndex}`);
        const ctx = canvas.getContext('2d');

        // Destroy existing chart
        if (detailCharts.has(cellIndex)) {
            detailCharts.get(cellIndex).destroy();
        }

        // Get experiment data
        const exp = experiments.get(expId);
        if (!exp) {
            const chart = createPlaceholderChart(ctx, 'No data available');
            detailCharts.set(cellIndex, chart);
            return;
        }

        console.log(`📊 Creating ${chartType} chart with real data for ${exp.name}`);
        console.log('Available data:', {
            semantic: !!exp.semantic,
            severity: !!exp.severity,
            summary: !!exp.summary
        });

        let chart;
        
        switch (chartType) {
            case 'stats-summary':
                chart = createStatsSummaryFixed(ctx, exp);
                break;
            case 'semantic-histogram':
                chart = createSemanticHistogramFixed(ctx, exp);
                break;
            case 'severity-levels':
                chart = createSeverityLevelsChartFixed(ctx, exp);
                break;
            case 'optimist-pessimist':
                chart = createOptimistPessimistChartFixed(ctx, exp);
                break;
            case 'semantic-bias':
                chart = createSemanticBiasChart(ctx, exp); // This one already uses real data
                break;
            case 'ridge-plot':
                chart = createRidgePlotFixed(ctx, exp);
                break;
            case 'gdx-ddx-severity':
                chart = createGDXvsDDXChartFixed(ctx, exp);
                break;
            default:
                chart = createPlaceholderChart(ctx, `Chart type: ${chartType}`);
        }

        detailCharts.set(cellIndex, chart);
    };
    
    // Fixed ridge plot using real data
    window.createRidgePlotFixed = function(ctx, exp) {
        if (!exp.semantic || !exp.severity) {
            console.warn('Missing semantic or severity data for ridge plot');
            return createPlaceholderChart(ctx, 'Missing data for ridge plot');
        }
        
        const scoresByGDXSeverity = {};
        for (let i = 3; i <= 10; i++) {
            scoresByGDXSeverity[`S${i}`] = [];
        }
        
        // Use real data: group semantic scores by GDX severity
        const semanticEvals = exp.semantic.evaluations || [];
        const severityEvals = exp.severity.evaluations || [];
        
        // Create lookup map
        const severityMap = new Map();
        severityEvals.forEach(sev => {
            severityMap.set(sev.id, sev);
        });
        
        semanticEvals.forEach((semEval, idx) => {
            const caseId = semEval.case_id || idx + 1;
            const severityEval = severityMap.get(caseId);
            
            if (severityEval && severityEval.gdx) {
                const gdxSeverity = severityEval.gdx.severity;
                const semanticScore = semEval.best_match?.score || 0;
                
                if (scoresByGDXSeverity[gdxSeverity]) {
                    scoresByGDXSeverity[gdxSeverity].push(semanticScore);
                }
            }
        });
        
        // Calculate KDE for each severity level
        const ridgeData = [];
        const severityLevels = Object.keys(scoresByGDXSeverity).reverse();
        
        severityLevels.forEach((severity, idx) => {
            const scores = scoresByGDXSeverity[severity];
            if (scores.length > 0) {
                const kde = calculateKDE(scores, 0.05, 0, 1, 100);
                const maxDensity = Math.max(...kde.map(d => d.y));
                
                const normalizedKDE = kde.map(point => ({
                    x: point.x,
                    y: (point.y / maxDensity) * 0.8 + idx
                }));
                
                ridgeData.push({
                    severity: severity,
                    kde: normalizedKDE,
                    baseline: idx,
                    color: `hsl(${260 - idx * 20}, 70%, 60%)`,
                    count: scores.length
                });
            }
        });
        
        // Ridge plot plugin (same as before but with real data)
        const ridgePlugin = {
            id: 'ridgeReal',
            beforeDraw: (chart) => {
                const ctx = chart.ctx;
                const chartArea = chart.chartArea;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                
                ridgeData.forEach((ridge, idx) => {
                    // Draw filled area
                    ctx.save();
                    ctx.beginPath();
                    
                    // Start from baseline
                    ctx.moveTo(
                        xScale.getPixelForValue(0),
                        yScale.getPixelForValue(ridge.baseline)
                    );
                    
                    // Draw KDE curve
                    ridge.kde.forEach(point => {
                        const x = xScale.getPixelForValue(point.x);
                        const y = yScale.getPixelForValue(point.y);
                        ctx.lineTo(x, y);
                    });
                    
                    // Close path at baseline
                    ctx.lineTo(
                        xScale.getPixelForValue(1),
                        yScale.getPixelForValue(ridge.baseline)
                    );
                    ctx.closePath();
                    
                    // Fill with gradient
                    const gradient = ctx.createLinearGradient(
                        0, yScale.getPixelForValue(ridge.baseline + 0.8),
                        0, yScale.getPixelForValue(ridge.baseline)
                    );
                    gradient.addColorStop(0, ridge.color);
                    gradient.addColorStop(1, ridge.color.replace('60%', '30%'));
                    
                    ctx.fillStyle = gradient;
                    ctx.globalAlpha = 0.7;
                    ctx.fill();
                    
                    // Draw outline
                    ctx.strokeStyle = ridge.color;
                    ctx.globalAlpha = 1;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // Add severity label with count
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 12px Inter';
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(
                        `${ridge.severity} (n=${ridge.count})`,
                        chartArea.left - 10,
                        yScale.getPixelForValue(ridge.baseline + 0.4)
                    );
                    
                    ctx.restore();
                });
            }
        };
        
        return new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    data: [] // Empty dataset for Chart.js structure
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
                        left: 80,
                        right: 20
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `Real Semantic Score Ridge Plot by GDX Severity`,
                        color: '#ffffff'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Semantic Score',
                            color: '#ffffff'
                        },
                        min: 0,
                        max: 1,
                        ticks: {
                            color: '#9ca3af'
                        },
                        grid: {
                            color: '#1f2937'
                        }
                    },
                    y: {
                        type: 'linear',
                        min: -0.5,
                        max: severityLevels.length - 0.5,
                        display: false
                    }
                }
            },
            plugins: [ridgePlugin]
        });
    };
    
    // Fixed GDX vs DDX chart using real data
    window.createGDXvsDDXChartFixed = function(ctx, exp) {
        if (!exp.severity) {
            console.warn('No severity data for GDX vs DDX chart');
            return createPlaceholderChart(ctx, 'No severity data available');
        }
        
        const gdxDdxData = [];
        const gdxCounts = {};
        
        // Process real severity evaluations
        exp.severity.evaluations.forEach(evaluation => {
            if (evaluation.gdx && evaluation.ddx_list) {
                const gdxSeverity = parseInt(evaluation.gdx.severity.replace('S', ''));
                
                if (!gdxCounts[gdxSeverity]) {
                    gdxCounts[gdxSeverity] = 0;
                }
                gdxCounts[gdxSeverity]++;
                
                evaluation.ddx_list.forEach(ddx => {
                    const ddxSeverity = parseInt(ddx.severity.replace('S', ''));
                    gdxDdxData.push({
                        gdx: gdxSeverity,
                        ddx: ddxSeverity,
                        distance: ddx.distance || Math.abs(gdxSeverity - ddxSeverity)
                    });
                });
            }
        });
        
        // Create perfect match line
        const perfectMatch = [];
        for (let i = 0; i <= 10; i++) {
            perfectMatch.push({x: i, y: i});
        }
        
        return new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
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
                    label: `Real GDX vs DDX (n=${gdxDdxData.length})`,
                    data: gdxDdxData.map(d => ({x: d.gdx, y: d.ddx})),
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    order: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Real GDX vs DDX Severity Comparison (${exp.severity.evaluations.length} cases)`,
                        color: '#ffffff'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'GDX Severity',
                            color: '#ffffff'
                        },
                        min: 0,
                        max: 10
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'DDX Severity',
                            color: '#ffffff'
                        },
                        min: 0,
                        max: 10
                    }
                }
            }
        });
    };
    
    console.log('✅ Dashboard data fix patch applied successfully!');
    console.log('📊 Charts will now use real experiment data instead of simulated data');
    
})(); 