#!/usr/bin/env python3
"""
Simple ICD10 Diversity Analyzer
Analyzes ICD10 code distribution in medical datasets
"""

import json
import yaml
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any

# Add utils to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.icd10.taxonomy import ICD10Taxonomy


def load_config():
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_icd10_codes(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract all ICD10 codes from dataset."""
    results = {
        'total_cases': len(data),
        'cases_with_icd10': 0,
        'cases_without_icd10': 0,
        'all_codes': [],
        'missing_cases': []
    }
    
    for item in data:
        found_code = False
        
        # Check all possible ICD10 fields
        for field in ['icd10', 'icd10_code', 'icd_code', 'diagnosis_code', 'diagnostic_code', 'code']:
            if field in item and item[field]:
                code = str(item[field]).strip()
                if code:
                    results['all_codes'].append(code)
                    found_code = True
                    break
        
        # Check diagnoses array
        if not found_code and 'diagnoses' in item and isinstance(item['diagnoses'], list):
            for diag in item['diagnoses']:
                if isinstance(diag, dict):
                    for field in ['icd10_code', 'code', 'diagnostic_code']:
                        if field in diag and diag[field]:
                            results['all_codes'].append(str(diag[field]).strip())
                            found_code = True
                            break
        
        # Check diagnosis field directly
        if not found_code and 'diagnosis' in item and isinstance(item['diagnosis'], str):
            # Check if there's a related code field
            for field in ['diagnosis_code', 'diagnostic_code', 'code']:
                if field in item and item[field]:
                    results['all_codes'].append(str(item[field]).strip())
                    found_code = True
                    break
        
        if found_code:
            results['cases_with_icd10'] += 1
        else:
            results['cases_without_icd10'] += 1
            results['missing_cases'].append({
                'id': item.get('id', 'unknown'),
                'diagnosis': item.get('diagnosis', item.get('diagnoses', 'No diagnosis found'))
            })
    
    return results


def analyze_icd10_hierarchy(codes: List[str]) -> Dict[str, Any]:
    """Analyze ICD10 codes across the taxonomy hierarchy."""
    taxonomy = ICD10Taxonomy()
    
    analysis = {
        'total_codes': len(codes),
        'unique_codes': len(set(codes)),
        'chapters': defaultdict(lambda: {'count': 0, 'codes': set(), 'name': ''}),
        'categories': defaultdict(int),
        'blocks': defaultdict(int),
        'invalid_codes': [],
        'resolved_codes': []
    }
    
    # Analyze each code
    for code in codes:
        if not code:
            continue
        
        # Clean the code
        code = code.strip().upper()
        
        # Remove common suffixes (A, B, D, S, etc.) that indicate encounter type
        base_code = code
        if len(code) > 7 and code[-1].isalpha() and code[-2] not in ['.', '-']:
            base_code = code[:-1]
        
        # Try to get info about the code using taxonomy
        code_info = taxonomy.get(base_code) or taxonomy.get(code)
        
        if code_info:
            # Valid code found
            analysis['resolved_codes'].append({
                'original': code,
                'resolved': code_info['code'],
                'name': code_info['name'],
                'type': code_info['type']
            })
            
            # Get full hierarchy
            hierarchy = taxonomy.hierarchy(code)
            
            if hierarchy:
                # Chapter level
                if 'parents' in hierarchy and 'chapter' in hierarchy['parents']:
                    chapter = hierarchy['parents']['chapter']
                    chapter_code = chapter['code']
                    analysis['chapters'][chapter_code]['count'] += 1
                    analysis['chapters'][chapter_code]['codes'].add(code)
                    analysis['chapters'][chapter_code]['name'] = chapter['name']
                elif hierarchy['type'] == 'chapter':
                    # This IS a chapter
                    analysis['chapters'][code]['count'] += 1
                    analysis['chapters'][code]['codes'].add(code)
                    analysis['chapters'][code]['name'] = code_info['name']
                
                # Get category level
                if 'parents' in hierarchy and 'category' in hierarchy['parents']:
                    category = hierarchy['parents']['category']['code']
                    analysis['categories'][category] += 1
                elif hierarchy['type'] == 'category' or (len(code) >= 3 and code[:3] in taxonomy._code_map):
                    category = code if hierarchy['type'] == 'category' else code[:3]
                    analysis['categories'][category] += 1
                
                # Count blocks and subblocks
                if hierarchy['type'] in ['block', 'subblock', 'group', 'subgroup']:
                    analysis['blocks'][code] += 1
        else:
            # Code not found directly, try to extract category
            if len(code) >= 3:
                category_attempt = code[:3]
                if taxonomy.get(category_attempt):
                    analysis['categories'][category_attempt] += 1
                    # Get chapter for this category
                    cat_hierarchy = taxonomy.hierarchy(category_attempt)
                    if cat_hierarchy and 'parents' in cat_hierarchy and 'chapter' in cat_hierarchy['parents']:
                        chapter = cat_hierarchy['parents']['chapter']
                        chapter_code = chapter['code']
                        analysis['chapters'][chapter_code]['count'] += 1
                        analysis['chapters'][chapter_code]['codes'].add(code)
                        analysis['chapters'][chapter_code]['name'] = chapter['name']
                else:
                    analysis['invalid_codes'].append(code)
            else:
                analysis['invalid_codes'].append(code)
    
    # Convert chapters to list and calculate percentages
    total = len(codes)
    chapter_list = []
    for chapter_code, data in analysis['chapters'].items():
        chapter_list.append({
            'code': chapter_code,
            'name': data['name'],
            'count': data['count'],
            'percentage': round(data['count'] / total * 100, 2) if total > 0 else 0,
            'unique_codes': len(data['codes'])
        })
    
    analysis['chapters'] = sorted(chapter_list, key=lambda x: x['count'], reverse=True)
    
    return analysis


def build_hierarchy_tree(codes: List[str], taxonomy) -> Dict[str, Any]:
    """Build a hierarchical tree structure for visualization."""
    tree = {
        'name': 'ICD10',
        'value': len(codes),
        'children': {}
    }
    
    for code in codes:
        if not code:
            continue
            
        code = code.strip().upper()
        # Remove suffix
        if len(code) > 7 and code[-1].isalpha() and code[-2] not in ['.', '-']:
            code = code[:-1]
            
        # Get hierarchy
        hierarchy = taxonomy.hierarchy(code)
        if not hierarchy:
            continue
            
        # Build path through tree
        current = tree
        
        # Chapter level
        if 'parents' in hierarchy and 'chapter' in hierarchy['parents']:
            chapter = hierarchy['parents']['chapter']
            ch_key = f"{chapter['code']}: {chapter['name']}"
            if ch_key not in current['children']:
                current['children'][ch_key] = {
                    'name': ch_key,
                    'code': chapter['code'],
                    'value': 0,
                    'children': {}
                }
            current['children'][ch_key]['value'] += 1
            current = current['children'][ch_key]
            
            # Category level
            if 'category' in hierarchy['parents']:
                category = hierarchy['parents']['category']
                cat_key = f"{category['code']}: {category['name'][:50]}"
                if cat_key not in current['children']:
                    current['children'][cat_key] = {
                        'name': cat_key,
                        'code': category['code'],
                        'value': 0,
                        'children': {}
                    }
                current['children'][cat_key]['value'] += 1
                current = current['children'][cat_key]
                
                # Block level (if exists)
                if 'block' in hierarchy['parents']:
                    block = hierarchy['parents']['block']
                    block_key = f"{block['code']}: {block['name'][:40]}"
                    if block_key not in current['children']:
                        current['children'][block_key] = {
                            'name': block_key,
                            'code': block['code'],
                            'value': 0,
                            'children': {}
                        }
                    current['children'][block_key]['value'] += 1
    
    # Convert to list format
    def dict_to_list(node):
        if node['children']:
            node['children'] = [dict_to_list(child) for child in node['children'].values()]
        else:
            del node['children']
        return node
    
    return dict_to_list(tree)


def generate_html_report(dataset_name: str, extraction_results: Dict, hierarchy_analysis: Dict) -> str:
    """Generate an interactive HTML report with charts and drill-down capabilities."""
    
    # Calculate key metrics
    total_cases = extraction_results['total_cases']
    with_codes = extraction_results['cases_with_icd10']
    without_codes = extraction_results['cases_without_icd10']
    coverage_percent = round(with_codes / total_cases * 100, 1) if total_cases > 0 else 0
    
    # Build hierarchy tree for interactive visualization
    taxonomy = ICD10Taxonomy()
    hierarchy_tree = build_hierarchy_tree(extraction_results['all_codes'], taxonomy)
    
    # Generate chapter rows with status badges
    chapter_rows = ""
    for ch in hierarchy_analysis['chapters']:
        chapter_name = f"{ch['code']}: {ch['name']}" if ch['name'] else ch['code']
        
        # Add status badge
        if ch['percentage'] > 10:
            status = '<span class="badge badge-success">Alto</span>'
        elif ch['percentage'] > 5:
            status = '<span class="badge badge-warning">Medio</span>'
        else:
            status = '<span class="badge badge-danger">Bajo</span>'
            
        chapter_rows += f"""
        <tr>
            <td>{chapter_name}</td>
            <td class="text-right">{ch['count']:,}</td>
            <td class="text-right">{ch['percentage']}%</td>
            <td class="text-right">{ch['unique_codes']}</td>
            <td class="text-right">{status}</td>
        </tr>
        """
    
    # Generate diversity metrics
    num_chapters = len(hierarchy_analysis['chapters'])
    num_categories = len(hierarchy_analysis['categories'])
    diversity_score = "Alta" if num_chapters > 10 else "Media" if num_chapters > 5 else "Baja"
    diversity_color = "#28a745" if num_chapters > 10 else "#ffc107" if num_chapters > 5 else "#dc3545"
    
    # Prepare data for charts
    chapter_data = json.dumps([{"name": ch['name'], "value": ch['count']} for ch in hierarchy_analysis['chapters'][:10]])
    tree_data = json.dumps(hierarchy_tree)
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análisis ICD10 - {dataset_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
            color: white;
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .subtitle {{
            margin-top: 10px;
            opacity: 0.9;
        }}
        .tabs {{
            background: white;
            padding: 0;
            display: flex;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .tab {{
            padding: 20px 30px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
            font-weight: 500;
            color: #666;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }}
        .tab:hover {{
            color: #0066cc;
            background: rgba(0,102,204,0.05);
        }}
        .tab.active {{
            color: #0066cc;
            border-bottom-color: #0066cc;
        }}
        .content {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .tab-content {{
            display: none;
            animation: fadeIn 0.3s;
        }}
        .tab-content.active {{
            display: block;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            text-align: center;
            transition: transform 0.3s;
        }}
        .metric:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 3em;
            font-weight: bold;
            color: #0066cc;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 1em;
        }}
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
        }}
        .chart-container canvas {{
            max-height: 400px !important;
            display: block !important;
        }}
        .chart-title {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        @media (max-width: 968px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        #sunburstChart {{
            width: 100%;
            height: 600px;
        }}
        .breadcrumb {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
            color: #666;
        }}
        .breadcrumb span {{
            color: #0066cc;
            font-weight: 500;
        }}
        .info-panel {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
        }}
        .info-row:last-child {{
            border-bottom: none;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .text-right {{
            text-align: right;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}
        .diversity-indicator {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 25px;
            background: {diversity_color};
            color: white;
            font-weight: bold;
            font-size: 1.2em;
        }}
        .coverage-bar {{
            width: 100%;
            height: 40px;
            background: #e9ecef;
            border-radius: 20px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .coverage-fill {{
            height: 100%;
            background: {'#28a745' if coverage_percent > 80 else '#ffc107' if coverage_percent > 50 else '#dc3545'};
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            width: {coverage_percent}%;
            transition: width 1s ease;
        }}
        .filter-controls {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .filter-group {{
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        select {{
            padding: 8px 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            background: white;
            cursor: pointer;
        }}
        button {{
            padding: 8px 20px;
            border: none;
            border-radius: 5px;
            background: #0066cc;
            color: white;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.3s;
        }}
        button:hover {{
            background: #0052a3;
        }}
        .tooltip {{
            position: absolute;
            text-align: center;
            padding: 12px;
            font-size: 14px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 8px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Análisis de Diversidad ICD10</h1>
        <div class="subtitle">{dataset_name} - {total_cases:,} casos analizados</div>
    </div>
    
    <div class="tabs">
        <button class="tab active" onclick="showTab('overview')">Resumen</button>
        <button class="tab" onclick="showTab('charts')">Gráficos</button>
        <button class="tab" onclick="showTab('hierarchy')">Jerarquía Interactiva</button>
        <button class="tab" onclick="showTab('details')">Tabla Detallada</button>
        <button class="tab" onclick="showTab('analysis')">Análisis Profundo</button>
    </div>
    
    <div class="content">
        <!-- Tab 1: Overview -->
        <div id="overview" class="tab-content active">
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Total de Casos</div>
                    <div class="metric-value">{total_cases:,}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Con Códigos ICD10</div>
                    <div class="metric-value">{with_codes:,}</div>
                    <div class="metric-label">{coverage_percent}% cobertura</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Códigos Únicos</div>
                    <div class="metric-value">{hierarchy_analysis['unique_codes']:,}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Diversidad</div>
                    <div class="metric-value">
                        <span class="diversity-indicator">{diversity_score}</span>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2 class="chart-title">Cobertura de Códigos ICD10</h2>
                <div class="coverage-bar">
                    <div class="coverage-fill">{coverage_percent}%</div>
                </div>
                {'<div style="background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 8px; margin-top: 20px;"><strong>Atención:</strong> ' + str(without_codes) + ' casos no tienen códigos ICD10 asignados.</div>' if without_codes > 0 else ''}
            </div>
            
            <div class="chart-container">
                <h2 class="chart-title">Resumen de Diversidad</h2>
                <div class="info-panel">
                    <div class="info-row">
                        <span>Capítulos ICD10 cubiertos</span>
                        <strong>{num_chapters} de 22</strong>
                    </div>
                    <div class="info-row">
                        <span>Categorías únicas (3 caracteres)</span>
                        <strong>{num_categories}</strong>
                    </div>
                    <div class="info-row">
                        <span>Subcategorías detalladas</span>
                        <strong>{len(hierarchy_analysis['blocks'])}</strong>
                    </div>
                    <div class="info-row">
                        <span>Códigos no reconocidos</span>
                        <strong>{len(hierarchy_analysis['invalid_codes'])}</strong>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Tab 2: Charts -->
        <div id="charts" class="tab-content">
            <div class="chart-grid">
                <div class="chart-container">
                    <h2 class="chart-title">Distribución por Capítulos (Top 10)</h2>
                    <canvas id="chapterPieChart" style="max-width: 100%; height: 400px;"></canvas>
                </div>
                <div class="chart-container">
                    <h2 class="chart-title">Casos por Capítulo</h2>
                    <canvas id="chapterBarChart" style="max-width: 100%; height: 400px;"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h2 class="chart-title">Tendencia de Diversidad por Categorías</h2>
                <canvas id="categoryChart" style="max-width: 100%; height: 300px;"></canvas>
            </div>
        </div>
        
        <!-- Tab 3: Interactive Hierarchy -->
        <div id="hierarchy" class="tab-content">
            <div class="filter-controls">
                <div class="filter-group">
                    <label>Nivel de detalle:</label>
                    <select id="depthSelector" onchange="updateSunburst()">
                        <option value="2">Capítulos y Categorías</option>
                        <option value="3" selected>Hasta Bloques</option>
                        <option value="4">Detalle Completo</option>
                    </select>
                    <button onclick="resetZoom()">Restablecer Vista</button>
                </div>
            </div>
            
            <div class="breadcrumb" id="breadcrumb">
                <span>ICD10</span> → Haz clic en cualquier sección para explorar
            </div>
            
            <div class="chart-container">
                <h2 class="chart-title">Exploración Jerárquica Interactiva</h2>
                <div id="sunburstChart"></div>
                <div class="info-panel" id="selectionInfo" style="display: none;">
                    <h3 id="selectionTitle"></h3>
                    <div id="selectionDetails"></div>
                </div>
            </div>
        </div>
        
        <!-- Tab 4: Detailed Table -->
        <div id="details" class="tab-content">
            <div class="chart-container">
                <h2 class="chart-title">Tabla Detallada por Capítulos</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Capítulo</th>
                            <th class="text-right">Casos</th>
                            <th class="text-right">%</th>
                            <th class="text-right">Códigos Únicos</th>
                            <th class="text-right">Estado</th>
                        </tr>
                    </thead>
                    <tbody>
                        {chapter_rows if chapter_rows else '<tr><td colspan="5" style="text-align: center; color: #999;">No hay datos de capítulos ICD10</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Tab 5: Deep Analysis -->
        <div id="analysis" class="tab-content">
            <div class="chart-container">
                <h2 class="chart-title">Análisis de Concentración</h2>
                <canvas id="concentrationChart" style="max-width: 100%; height: 400px;"></canvas>
            </div>
            
            <div class="chart-container">
                <h2 class="chart-title">Matriz de Especialidades Médicas</h2>
                <div id="specialtyMatrix"></div>
            </div>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Global data
        const chapterData = {chapter_data};
        const hierarchyTree = {tree_data};
        
        // Tab switching
        function showTab(tabId) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
            
            // Initialize charts when shown
            if (tabId === 'charts' && !window.chartsInitialized) {{
                initializeCharts();
                window.chartsInitialized = true;
            }} else if (tabId === 'hierarchy' && !window.sunburstInitialized) {{
                initializeSunburst();
                window.sunburstInitialized = true;
            }} else if (tabId === 'analysis' && !window.analysisInitialized) {{
                initializeAnalysis();
                window.analysisInitialized = true;
            }}
        }}
        
        // Initialize Charts
        function initializeCharts() {{
            // Pie Chart
            const pieCtx = document.getElementById('chapterPieChart').getContext('2d');
            new Chart(pieCtx, {{
                type: 'doughnut',
                data: {{
                    labels: chapterData.map(d => d.name.split(':')[0]),
                    datasets: [{{
                        data: chapterData.map(d => d.value),
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                            '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384'
                        ],
                        hoverOffset: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'right',
                            labels: {{
                                padding: 15,
                                font: {{ size: 12 }}
                            }}
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const label = context.label || '';
                                    const value = context.parsed;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = ((value / total) * 100).toFixed(1);
                                    return label + ': ' + value + ' (' + percentage + '%)';
                                }}
                            }}
                        }}
                    }}
                }}
            }});
            
            // Bar Chart
            const barCtx = document.getElementById('chapterBarChart').getContext('2d');
            new Chart(barCtx, {{
                type: 'bar',
                data: {{
                    labels: chapterData.map(d => d.name.split(':')[0]),
                    datasets: [{{
                        label: 'Número de Casos',
                        data: chapterData.map(d => d.value),
                        backgroundColor: 'rgba(0, 102, 204, 0.8)',
                        borderColor: 'rgba(0, 102, 204, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return 'Casos: ' + context.parsed.x;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            beginAtZero: true,
                            grid: {{ display: true }}
                        }},
                        y: {{
                            grid: {{ display: false }}
                        }}
                    }}
                }}
            }});
            
            // Category trend chart
            const catCtx = document.getElementById('categoryChart').getContext('2d');
            const categoryLabels = chapterData.map(d => d.name.split(':')[0]);
            const cumulativeData = chapterData.map((d, i) => 
                chapterData.slice(0, i + 1).reduce((sum, item) => sum + item.value, 0)
            );
            
            new Chart(catCtx, {{
                type: 'line',
                data: {{
                    labels: categoryLabels,
                    datasets: [{{
                        label: 'Casos Acumulados',
                        data: cumulativeData,
                        borderColor: 'rgb(0, 102, 204)',
                        backgroundColor: 'rgba(0, 102, 204, 0.1)',
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{ color: 'rgba(0, 0, 0, 0.05)' }}
                        }},
                        x: {{
                            grid: {{ display: false }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Initialize Sunburst
        function initializeSunburst() {{
            const width = 800;
            const height = 600;
            const radius = Math.min(width, height) / 2;
            
            const svg = d3.select("#sunburstChart")
                .append("svg")
                .attr("viewBox", `0 0 ${{width}} ${{height}}`)
                .attr("width", "100%")
                .attr("height", "100%");
            
            const g = svg.append("g")
                .attr("transform", `translate(${{width/2}},${{height/2}})`);
            
            const partition = d3.partition()
                .size([2 * Math.PI, radius]);
            
            const arc = d3.arc()
                .startAngle(d => d.x0)
                .endAngle(d => d.x1)
                .innerRadius(d => d.y0)
                .outerRadius(d => d.y1);
            
            const color = d3.scaleOrdinal(d3.schemeCategory10);
            
            const root = d3.hierarchy(hierarchyTree)
                .sum(d => d.value)
                .sort((a, b) => b.value - a.value);
            
            partition(root);
            
            const path = g.selectAll("path")
                .data(root.descendants())
                .enter().append("path")
                .attr("d", arc)
                .style("fill", d => color((d.children ? d : d.parent).data.name))
                .style("stroke", "#fff")
                .style("stroke-width", 2)
                .style("cursor", "pointer")
                .on("click", clicked)
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);
            
            function clicked(event, p) {{
                const t = g.transition().duration(750);
                
                // Update breadcrumb
                const sequence = p.ancestors().reverse();
                const breadcrumb = sequence.map(d => d.data.name).join(" → ");
                document.getElementById("breadcrumb").innerHTML = `<span>${{breadcrumb}}</span>`;
                
                // Update selection info
                updateSelectionInfo(p);
                
                // Zoom animation
                root.each(d => d.target = {{
                    x0: Math.max(0, Math.min(1, (d.x0 - p.x0) / (p.x1 - p.x0))) * 2 * Math.PI,
                    x1: Math.max(0, Math.min(1, (d.x1 - p.x0) / (p.x1 - p.x0))) * 2 * Math.PI,
                    y0: Math.max(0, d.y0 - p.y0),
                    y1: Math.max(0, d.y1 - p.y0)
                }});
                
                path.transition(t)
                    .tween("data", d => {{
                        const i = d3.interpolate(d.current, d.target);
                        return t => d.current = i(t);
                    }})
                    .attrTween("d", d => () => arc(d.current));
            }}
            
            function showTooltip(event, d) {{
                const tooltip = document.getElementById("tooltip");
                tooltip.innerHTML = `
                    <strong>${{d.data.name}}</strong><br>
                    Casos: ${{d.value}}<br>
                    ${{((d.value / root.value) * 100).toFixed(1)}}% del total
                `;
                tooltip.style.opacity = 1;
                tooltip.style.left = (event.pageX + 10) + "px";
                tooltip.style.top = (event.pageY - 28) + "px";
            }}
            
            function hideTooltip() {{
                document.getElementById("tooltip").style.opacity = 0;
            }}
            
            window.resetZoom = function() {{
                clicked(null, root);
            }};
            
            path.each(function(d) {{ d.current = d; }});
        }}
        
        function updateSelectionInfo(node) {{
            const panel = document.getElementById("selectionInfo");
            const title = document.getElementById("selectionTitle");
            const details = document.getElementById("selectionDetails");
            
            panel.style.display = "block";
            title.textContent = node.data.name;
            
            const percentage = ((node.value / hierarchyTree.value) * 100).toFixed(1);
            const children = node.children ? node.children.length : 0;
            
            details.innerHTML = `
                <div class="info-row">
                    <span>Casos totales</span>
                    <strong>${{node.value}}</strong>
                </div>
                <div class="info-row">
                    <span>Porcentaje del total</span>
                    <strong>${{percentage}}%</strong>
                </div>
                <div class="info-row">
                    <span>Subcategorías</span>
                    <strong>${{children}}</strong>
                </div>
            `;
        }}
        
        function initializeAnalysis() {{
            // Concentration chart (Pareto)
            const concCtx = document.getElementById('concentrationChart').getContext('2d');
            const sortedData = [...chapterData].sort((a, b) => b.value - a.value);
            const total = sortedData.reduce((sum, d) => sum + d.value, 0);
            let cumulative = 0;
            const cumulativePercentages = sortedData.map(d => {{
                cumulative += d.value;
                return (cumulative / total) * 100;
            }});
            
            new Chart(concCtx, {{
                type: 'bar',
                data: {{
                    labels: sortedData.map(d => d.name.split(':')[0]),
                    datasets: [{{
                        type: 'bar',
                        label: 'Casos',
                        data: sortedData.map(d => d.value),
                        backgroundColor: 'rgba(0, 102, 204, 0.8)',
                        yAxisID: 'y'
                    }}, {{
                        type: 'line',
                        label: '% Acumulado',
                        data: cumulativePercentages,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        yAxisID: 'y1'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Análisis de Pareto - Concentración de Casos'
                        }}
                    }},
                    scales: {{
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            beginAtZero: true
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            beginAtZero: true,
                            max: 100,
                            grid: {{ drawOnChartArea: false }},
                            ticks: {{
                                callback: function(value) {{
                                    return value + '%';
                                }}
                            }}
                        }}
                    }}
                }}
            }});
            
            // Specialty matrix
            createSpecialtyMatrix();
        }}
        
        function createSpecialtyMatrix() {{
            const specialties = {{
                'Trauma/Urgencias': ['S', 'T', 'V', 'W', 'X', 'Y'],
                'Medicina Interna': ['I', 'J', 'K'],
                'Neurología': ['G', 'F'],
                'Pediatría': ['P', 'Q'],
                'Infectología': ['A', 'B'],
                'Oncología': ['C', 'D'],
                'Otros': ['E', 'H', 'L', 'M', 'N', 'O', 'R', 'U', 'Z']
            }};
            
            const matrixData = [];
            Object.entries(specialties).forEach(([specialty, chapters]) => {{
                const count = chapterData.filter(d => 
                    chapters.includes(d.name.split(':')[0])
                ).reduce((sum, d) => sum + d.value, 0);
                
                matrixData.push({{
                    specialty: specialty,
                    count: count,
                    percentage: ((count / {total_cases}) * 100).toFixed(1)
                }});
            }});
            
            const matrixHtml = `
                <table style="width: 100%;">
                    <thead>
                        <tr>
                            <th>Especialidad Médica</th>
                            <th class="text-right">Casos</th>
                            <th class="text-right">Porcentaje</th>
                            <th>Distribución</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${{matrixData.map(d => `
                            <tr>
                                <td>${{d.specialty}}</td>
                                <td class="text-right">${{d.count}}</td>
                                <td class="text-right">${{d.percentage}}%</td>
                                <td>
                                    <div style="background: #e9ecef; height: 20px; border-radius: 10px;">
                                        <div style="background: #0066cc; height: 100%; width: ${{d.percentage}}%; border-radius: 10px;"></div>
                                    </div>
                                </td>
                            </tr>
                        `).join('')}}
                    </tbody>
                </table>
            `;
            
            document.getElementById('specialtyMatrix').innerHTML = matrixHtml;
        }}
        
        // Initialize first tab on load
        window.addEventListener('load', function() {{
            // Charts will be initialized when tab is clicked
        }});
    </script>
</body>
</html>"""
    
    return html


def main():
    """Main analysis function."""
    # Load config
    config = load_config()
    dataset_path = config['dataset']
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Get dataset name
    dataset_name = Path(dataset_path).stem
    
    print(f"Analizando: {dataset_path}")
    
    # Load dataset
    data = load_dataset(dataset_path)
    print(f"Casos cargados: {len(data)}")
    
    # Extract ICD10 codes
    extraction = extract_icd10_codes(data)
    print(f"Casos con ICD10: {extraction['cases_with_icd10']}")
    print(f"Casos sin ICD10: {extraction['cases_without_icd10']}")
    
    # Analyze hierarchy if we have codes
    if extraction['all_codes']:
        print(f"Analizando {len(extraction['all_codes'])} códigos...")
        print(f"Primeros 5 códigos: {extraction['all_codes'][:5]}")
        hierarchy = analyze_icd10_hierarchy(extraction['all_codes'])
        print(f"Códigos únicos: {hierarchy['unique_codes']}")
        print(f"Capítulos encontrados: {len(hierarchy['chapters'])}")
        if hierarchy['invalid_codes']:
            print(f"Códigos no reconocidos: {len(hierarchy['invalid_codes'])}")
            print(f"  Ejemplos: {hierarchy['invalid_codes'][:5]}")
    else:
        hierarchy = {
            'total_codes': 0,
            'unique_codes': 0,
            'chapters': [],
            'categories': {},
            'blocks': {},
            'invalid_codes': []
        }
        print("ADVERTENCIA: No se encontraron códigos ICD10 en el dataset")
    
    # Generate HTML report
    html_content = generate_html_report(dataset_name, extraction, hierarchy)
    
    # Save report
    output_file = output_dir / f"{dataset_name}_icd10_analysis.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nAnálisis completado. Abre el archivo:")
    print(f"  {output_file}")


if __name__ == "__main__":
    main()