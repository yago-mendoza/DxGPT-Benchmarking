// Fixed data loading functions for experiment dashboard
// These functions replace the simulated data with real experiment data

// Extract real semantic scores from experiment data
function extractSemanticScores(exp) {
    if (!exp.semantic || !exp.semantic.evaluations) {
        console.warn('No semantic evaluation data found');
        return [];
    }
    
    return exp.semantic.evaluations.map(eval => {
        return eval.best_match?.score || 0;
    });
}

// Extract real severity scores from experiment data
function extractSeverityScores(exp) {
    if (!exp.severity || !exp.severity.evaluations) {
        console.warn('No severity evaluation data found');
        return [];
    }
    
    return exp.severity.evaluations.map(eval => {
        return eval.final_score || 0;
    });
}

// Fixed createStatsSummary - uses real data
function createStatsSummaryFixed(ctx, exp) {
    // Extract real semantic and severity scores
    const semanticScores = extractSemanticScores(exp);
    const severityScores = extractSeverityScores(exp);
    
    if (semanticScores.length === 0 || severityScores.length === 0) {
        console.warn('No data available for stats summary');
        return createPlaceholderChart(ctx, 'No data available');
    }
    
    const semanticMean = semanticScores.reduce((a, b) => a + b, 0) / semanticScores.length;
    const severityMean = severityScores.reduce((a, b) => a + b, 0) / severityScores.length;
    
    // Calculate additional statistics
    const semanticStd = Math.sqrt(semanticScores.reduce((sum, score) => sum + Math.pow(score - semanticMean, 2), 0) / semanticScores.length);
    const severityStd = Math.sqrt(severityScores.reduce((sum, score) => sum + Math.pow(score - severityMean, 2), 0) / severityScores.length);
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Semantic Mean', 'Severity Mean', 'Semantic Std', 'Severity Std'],
            datasets: [{
                label: 'Statistics',
                data: [semanticMean, severityMean, semanticStd, severityStd],
                backgroundColor: [
                    'rgba(99, 102, 241, 0.6)', 'rgba(239, 68, 68, 0.6)',
                    'rgba(99, 102, 241, 0.3)', 'rgba(239, 68, 68, 0.3)'
                ],
                borderColor: [
                    'rgba(99, 102, 241, 1)', 'rgba(239, 68, 68, 1)',
                    'rgba(99, 102, 241, 1)', 'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Real Data (${semanticScores.length} semantic, ${severityScores.length} severity evaluations)`,
                    color: '#ffffff'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// Fixed createSemanticHistogram - uses real data
function createSemanticHistogramFixed(ctx, exp) {
    const allSemanticScores = extractSemanticScores(exp);
    
    if (allSemanticScores.length === 0) {
        console.warn('No semantic scores available');
        return createPlaceholderChart(ctx, 'No semantic data available');
    }
    
    // Create histogram bins
    const bins = 25;
    const binWidth = 1.0 / bins;
    const histogram = new Array(bins).fill(0);
    
    allSemanticScores.forEach(score => {
        const binIndex = Math.min(Math.floor(score / binWidth), bins - 1);
        histogram[binIndex]++;
    });
    
    // Create bin labels
    const binLabels = [];
    const histogramData = [];
    for (let i = 0; i < bins; i++) {
        const binStart = i * binWidth;
        const binEnd = (i + 1) * binWidth;
        binLabels.push(`${binStart.toFixed(2)}-${binEnd.toFixed(2)}`);
        histogramData.push(histogram[i]);
    }
    
    // Calculate KDE using real data
    const kde = calculateKDE(allSemanticScores, 0.05, 0, 1, bins);
    const maxHistCount = Math.max(...histogram);
    const maxKDE = Math.max(...kde.map(d => d.y));
    
    // Scale KDE to match histogram height
    const kdeScaled = kde.map((point, i) => ({
        x: binLabels[Math.floor(i * bins / kde.length)] || binLabels[binLabels.length - 1],
        y: (point.y / maxKDE) * maxHistCount * 1.1
    }));

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: `Semantic Score Count (n=${allSemanticScores.length})`,
                data: histogramData,
                backgroundColor: 'rgba(99, 102, 241, 0.6)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 1,
                order: 2,
                barPercentage: 0.95,
                categoryPercentage: 1.0
            }, {
                label: 'KDE Curve (Real Data)',
                data: kdeScaled,
                type: 'line',
                borderColor: 'rgba(245, 158, 11, 0.9)',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                borderWidth: 3,
                fill: false,
                pointRadius: 0,
                tension: 0.4,
                borderCapStyle: 'round',
                borderJoinStyle: 'round',
                order: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Real Semantic Score Distribution (μ=${(allSemanticScores.reduce((a,b) => a+b, 0)/allSemanticScores.length).toFixed(3)})`,
                    color: '#ffffff'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Semantic Similarity Score Range'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Count / Density'
                    }
                }
            }
        }
    });
}

// Fixed severity levels chart - uses real data
function createSeverityLevelsChartFixed(ctx, exp) {
    if (!exp.severity || !exp.severity.evaluations) {
        console.warn('No severity evaluation data found');
        return createPlaceholderChart(ctx, 'No severity data available');
    }
    
    const gdxCounts = {};
    const ddxCounts = {};
    
    // Initialize counts
    for (let i = 3; i <= 10; i++) {
        gdxCounts[`S${i}`] = 0;
        ddxCounts[`S${i}`] = 0;
    }
    
    // Count real severity levels
    exp.severity.evaluations.forEach(evaluation => {
        // Count GDX severity
        const gdxSeverity = evaluation.gdx?.severity;
        if (gdxSeverity && gdxCounts[gdxSeverity] !== undefined) {
            gdxCounts[gdxSeverity]++;
        }
        
        // Count DDX severities
        if (evaluation.ddx_list) {
            evaluation.ddx_list.forEach(ddx => {
                const ddxSeverity = ddx.severity;
                if (ddxSeverity && ddxCounts[ddxSeverity] !== undefined) {
                    ddxCounts[ddxSeverity]++;
                }
            });
        }
    });
    
    // Prepare data for chart
    const severityLabels = [];
    const gdxData = [];
    const ddxData = [];
    
    for (let i = 3; i <= 10; i++) {
        const severity = `S${i}`;
        severityLabels.push(severity);
        gdxData.push(gdxCounts[severity]);
        ddxData.push(ddxCounts[severity]);
    }
    
    const totalGdx = gdxData.reduce((a, b) => a + b, 0);
    const totalDdx = ddxData.reduce((a, b) => a + b, 0);

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: severityLabels,
            datasets: [{
                label: `GDX (n=${totalGdx})`,
                data: gdxData,
                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 1,
                order: 3,
                barPercentage: 0.8,
                categoryPercentage: 0.9
            }, {
                label: `DDX (n=${totalDdx})`,
                data: ddxData,
                backgroundColor: 'rgba(249, 115, 22, 0.6)',
                borderColor: 'rgba(249, 115, 22, 1)',
                borderWidth: 1,
                order: 3,
                barPercentage: 0.8,
                categoryPercentage: 0.9
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Real Severity Level Distribution (${exp.severity.evaluations.length} cases)`,
                    color: '#ffffff'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Severity Level'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Count'
                    }
                }
            }
        }
    });
}

// Fixed optimist vs pessimist chart - uses real data
function createOptimistPessimistChartFixed(ctx, exp) {
    if (!exp.severity || !exp.severity.evaluations) {
        console.warn('No severity evaluation data found');
        return createPlaceholderChart(ctx, 'No severity data available');
    }
    
    let optimistCases = 0;
    let pessimistCases = 0;
    let neutralCases = 0;
    
    // Count by n value for each type
    const optimistByN = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0};
    const pessimistByN = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0};
    
    // Use real evaluation data
    exp.severity.evaluations.forEach(evaluation => {
        const optimistN = evaluation.optimist?.n || 0;
        const pessimistN = evaluation.pessimist?.n || 0;
        
        if (optimistN > pessimistN) {
            optimistCases++;
            const n = Math.min(5, Math.max(1, optimistN));
            optimistByN[n]++;
        } else if (pessimistN > optimistN) {
            pessimistCases++;
            const n = Math.min(5, Math.max(1, pessimistN));
            pessimistByN[n]++;
        } else {
            neutralCases++;
        }
    });

    // Create plugin to draw center text with real data
    const centerTextPlugin = {
        id: 'centerTextReal',
        beforeDraw: function(chart) {
            const ctx = chart.ctx;
            const centerX = (chart.chartArea.left + chart.chartArea.right) / 2;
            const centerY = (chart.chartArea.top + chart.chartArea.bottom) / 2;
            ctx.save();
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.font = '300 24px Inter';
            ctx.fillStyle = '#e5e7eb';
            ctx.fillText(optimistCases + ' | ' + pessimistCases, centerX, centerY - 15);
            ctx.font = '300 12px Inter';
            ctx.fillStyle = '#9ca3af';
            ctx.fillText('Optimist | Pessimist', centerX, centerY);
            ctx.fillText(`(${neutralCases} neutral)`, centerX, centerY + 15);
            ctx.restore();
        }
    };

    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [
                'Opt n=1', 'Opt n=2', 'Opt n=3', 'Opt n=4', 'Opt n=5',
                'Pes n=1', 'Pes n=2', 'Pes n=3', 'Pes n=4', 'Pes n=5'
            ],
            datasets: [{
                data: [
                    optimistByN[1], optimistByN[2], optimistByN[3], optimistByN[4], optimistByN[5],
                    pessimistByN[1], pessimistByN[2], pessimistByN[3], pessimistByN[4], pessimistByN[5]
                ],
                backgroundColor: [
                    // Optimist shades (blue gradient)
                    'rgba(59, 130, 246, 0.9)',
                    'rgba(59, 130, 246, 0.75)',
                    'rgba(59, 130, 246, 0.6)',
                    'rgba(59, 130, 246, 0.45)',
                    'rgba(59, 130, 246, 0.3)',
                    // Pessimist shades (red gradient)
                    'rgba(239, 68, 68, 0.9)',
                    'rgba(239, 68, 68, 0.75)',
                    'rgba(239, 68, 68, 0.6)',
                    'rgba(239, 68, 68, 0.45)',
                    'rgba(239, 68, 68, 0.3)'
                ],
                borderColor: '#1f2937',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Real Optimist/Pessimist Balance (${exp.severity.evaluations.length} cases)`,
                    color: '#ffffff'
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#ffffff',
                        font: { size: 12 },
                        padding: 15,
                        boxWidth: 15,
                        boxHeight: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed;
                            return label + ': ' + value + ' cases';
                        }
                    }
                }
            }
        },
        plugins: [centerTextPlugin]
    });
} 