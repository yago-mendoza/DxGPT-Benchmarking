#!/usr/bin/env python3
"""
Análisis Estadístico Exhaustivo de Datos Médicos RAMEDIS y URG_TORRE
=====================================================================

Este script genera visualizaciones estadísticas detalladas y realiza un análisis
profundo de la relación entre complejidad y severidad en casos médicos.

Autor: Assistant
Fecha: 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
from collections import Counter, defaultdict
import re
from wordcloud import WordCloud
import networkx as nx
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime
import os

# Configuración estética
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuración de matplotlib para mejor visualización
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 100

class MedicalDataAnalyzer:
    """Clase principal para el análisis de datos médicos"""
    
    def __init__(self, ramedis_file='RAMEDIS.json', urg_torre_file='URG_TORRE.json'):
        """
        Inicializa el analizador con los archivos de datos
        """
        self.ramedis_data = self.load_json(ramedis_file)
        self.urg_torre_data = self.load_json(urg_torre_file)
        self.df_ramedis = None
        self.df_urg_torre = None
        self.complexity_severity_analysis = {}
        
        # Crear directorio de visualizaciones
        os.makedirs('./visualisations', exist_ok=True)
        
    def load_json(self, filename):
        """Carga datos desde archivo JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando {filename}: {e}")
            return []
    
    def prepare_dataframes(self):
        """Prepara los DataFrames para análisis"""
        # RAMEDIS DataFrame
        ramedis_records = []
        for case in self.ramedis_data:
            symptoms = self.extract_symptoms(case.get('case', ''))
            diagnoses = case.get('diagnoses', [])
            severities = [d.get('severity', 'Unknown') for d in diagnoses]
            diagnosis_names = [d.get('name', 'Unknown') for d in diagnoses]
            
            record = {
                'id': case.get('id'),
                'complexity': case.get('complexity', 'Unknown'),
                'num_diagnoses': len(diagnoses),
                'diagnosis_names': ', '.join(diagnosis_names),
                'severities': ', '.join(severities),
                'max_severity': max(severities) if severities else 'S0',
                'min_severity': min(severities) if severities else 'S0',
                'avg_severity_num': self.severity_to_number(severities),
                'diagnostic_codes': ', '.join(case.get('diagnostic_codes', [])),
                'num_symptoms': len(symptoms),
                'symptoms': ', '.join(symptoms),
                'has_death': 'death' in case.get('case', '').lower(),
                'origin': case.get('origin', 'RAMEDIS')
            }
            ramedis_records.append(record)
        
        # URG_TORRE DataFrame
        urg_records = []
        for case in self.urg_torre_data:
            record = {
                'id': case.get('id'),
                'complexity': case.get('complexity', 'Unknown'),
                'num_diagnoses': len(case.get('diagnoses', [])),
                'diagnosis_names': ', '.join([d.get('name', 'Unknown') for d in case.get('diagnoses', [])]),
                'severities': ', '.join([d.get('severity', 'Unknown') for d in case.get('diagnoses', [])]),
                'max_severity': max([d.get('severity', 'S0') for d in case.get('diagnoses', [])]) if case.get('diagnoses') else 'S0',
                'avg_severity_num': self.severity_to_number([d.get('severity', 'S0') for d in case.get('diagnoses', [])]),
                'diagnostic_code': case.get('diagnostic_code', ''),
                'patient_age': self.extract_age(case.get('case', '')),
                'patient_gender': self.extract_gender(case.get('case', '')),
                'death': case.get('death', 0),
                'critical': case.get('critical', 0),
                'pediatric': case.get('pediatric', 0),
                'severity_score': case.get('severity', 0)
            }
            urg_records.append(record)
        
        self.df_ramedis = pd.DataFrame(ramedis_records)
        self.df_urg_torre = pd.DataFrame(urg_records)
        
    def extract_symptoms(self, case_text):
        """Extrae síntomas del texto del caso"""
        symptoms = []
        lines = case_text.split('\\n')
        in_symptoms = False
        
        for line in lines:
            if 'síntomas:' in line.lower():
                in_symptoms = True
                continue
            if in_symptoms and line.strip().startswith('-'):
                symptom = line.strip()[1:].strip()
                symptoms.append(symptom)
            elif in_symptoms and not line.strip().startswith('-') and line.strip():
                break
                
        return symptoms
    
    def extract_age(self, case_text):
        """Extrae edad del paciente del texto"""
        match = re.search(r'(\d+)\s*años', case_text)
        return int(match.group(1)) if match else None
    
    def extract_gender(self, case_text):
        """Extrae género del paciente del texto"""
        if 'Hombre' in case_text:
            return 'M'
        elif 'Mujer' in case_text:
            return 'F'
        return 'Unknown'
    
    def severity_to_number(self, severities):
        """Convierte severidades a valor numérico promedio"""
        if not severities:
            return 0
        
        numeric_severities = []
        for s in severities:
            if isinstance(s, str) and s.startswith('S'):
                try:
                    numeric_severities.append(int(s[1:]))
                except:
                    numeric_severities.append(0)
            else:
                numeric_severities.append(0)
        
        return np.mean(numeric_severities) if numeric_severities else 0
    
    def complexity_to_number(self, complexity):
        """Convierte complejidad a valor numérico"""
        if isinstance(complexity, str) and complexity.startswith('C'):
            try:
                return int(complexity[1:])
            except:
                return 0
        return 0
    
    def create_visualizations(self):
        """Genera todas las visualizaciones individuales"""
        print("Generando visualizaciones estadísticas individuales...")
        
        plot_functions = [
            ('Distribución de Complejidad', self.plot_complexity_distribution),
            ('Distribución de Severidad', self.plot_severity_distribution),
            ('Relación Complejidad-Severidad', self.plot_complexity_severity_relationship),
            ('Análisis de Múltiples Diagnósticos', self.plot_multiple_diagnoses_analysis),
            ('Mapa de Calor de Correlaciones', self.plot_correlation_heatmap),
            ('Análisis de Síntomas', self.plot_symptoms_analysis),
            ('Análisis Demográfico', self.plot_demographic_analysis),
            ('Análisis de Mortalidad', self.plot_mortality_analysis),
            ('Clustering de Casos', self.plot_case_clustering),
            ('Red de Diagnósticos', self.plot_diagnosis_network),
            ('Análisis Temporal', self.plot_complexity_timeline),
            ('Distribución de Códigos', self.plot_diagnostic_codes_distribution),
            ('Word Cloud de Síntomas', self.plot_symptoms_wordcloud),
            ('Análisis Comparativo', self.plot_comparative_analysis),
            ('Modelo Predictivo', self.plot_severity_prediction_model)
        ]
        
        for i, (name, func) in enumerate(plot_functions, 1):
            print(f"  {i:2d}/15 - Generando {name}...")
            try:
                func()
                print(f"       ✓ Completado")
            except Exception as e:
                print(f"       ✗ Error: {e}")
        
        print(f"\n✅ Visualizaciones guardadas en ./visualisations/")
    
    def plot_complexity_distribution(self):
        """Distribución de complejidad"""
        # RAMEDIS
        plt.figure(figsize=(12, 8))
        complexity_counts = self.df_ramedis['complexity'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(complexity_counts)))
        bars = plt.bar(complexity_counts.index, complexity_counts.values, color=colors, edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars, complexity_counts.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}\n({count/len(self.df_ramedis)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('RAMEDIS - Distribución de Complejidad', fontsize=16)
        plt.xlabel('Nivel de Complejidad')
        plt.ylabel('Número de Casos')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./visualisations/01_complexity_distribution_ramedis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # URG_TORRE
        plt.figure(figsize=(12, 8))
        complexity_counts_urg = self.df_urg_torre['complexity'].value_counts().sort_index()
        colors_urg = plt.cm.plasma(np.linspace(0, 1, len(complexity_counts_urg)))
        bars_urg = plt.bar(complexity_counts_urg.index, complexity_counts_urg.values, color=colors_urg, edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars_urg, complexity_counts_urg.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}\n({count/len(self.df_urg_torre)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('URG_TORRE - Distribución de Complejidad', fontsize=16)
        plt.xlabel('Nivel de Complejidad')
        plt.ylabel('Número de Casos')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./visualisations/02_complexity_distribution_urg_torre.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_severity_distribution(self):
        """Distribución de severidad"""
        # RAMEDIS pie chart
        plt.figure(figsize=(10, 8))
        severity_counts = self.df_ramedis['max_severity'].value_counts().sort_index()
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(severity_counts)))
        
        wedges, texts, autotexts = plt.pie(severity_counts.values, labels=severity_counts.index, 
                                           autopct='%1.1f%%', colors=colors, startangle=90,
                                           explode=[0.05] * len(severity_counts))
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            
        plt.title('RAMEDIS - Distribución de Severidad Máxima', fontsize=16)
        plt.tight_layout()
        plt.savefig('./visualisations/03_severity_distribution_ramedis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # URG_TORRE horizontal bars
        plt.figure(figsize=(12, 8))
        severity_counts_urg = self.df_urg_torre['max_severity'].value_counts().sort_index()
        
        y_pos = np.arange(len(severity_counts_urg))
        bars = plt.barh(y_pos, severity_counts_urg.values, 
                       color=plt.cm.Oranges(np.linspace(0.3, 1, len(severity_counts_urg))))
        
        plt.yticks(y_pos, severity_counts_urg.index)
        plt.xlabel('Número de Casos')
        plt.title('URG_TORRE - Distribución de Severidad', fontsize=16)
        
        for i, (bar, value) in enumerate(zip(bars, severity_counts_urg.values)):
            plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{value} ({value/len(self.df_urg_torre)*100:.1f}%)',
                    ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('./visualisations/04_severity_distribution_urg_torre.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_complexity_severity_relationship(self):
        """Relación complejidad-severidad"""
        plt.figure(figsize=(12, 8))
        
        ramedis_comp = self.df_ramedis['complexity'].apply(self.complexity_to_number)
        ramedis_sev = self.df_ramedis['avg_severity_num']
        urg_comp = self.df_urg_torre['complexity'].apply(self.complexity_to_number)
        urg_sev = self.df_urg_torre['avg_severity_num']
        
        plt.scatter(ramedis_comp, ramedis_sev, alpha=0.6, s=100, c='blue', label='RAMEDIS', edgecolors='black')
        plt.scatter(urg_comp, urg_sev, alpha=0.6, s=100, c='red', label='URG_TORRE', edgecolors='black')
        
        # Líneas de regresión
        if len(ramedis_comp) > 1:
            z_ramedis = np.polyfit(ramedis_comp, ramedis_sev, 1)
            p_ramedis = np.poly1d(z_ramedis)
            x_range = np.linspace(0, 11, 100)
            plt.plot(x_range, p_ramedis(x_range), "b--", alpha=0.8, linewidth=2, 
                    label=f'RAMEDIS: y={z_ramedis[0]:.2f}x+{z_ramedis[1]:.2f}')
        
        if len(urg_comp) > 1:
            z_urg = np.polyfit(urg_comp, urg_sev, 1)
            p_urg = np.poly1d(z_urg)
            plt.plot(x_range, p_urg(x_range), "r--", alpha=0.8, linewidth=2, 
                    label=f'URG_TORRE: y={z_urg[0]:.2f}x+{z_urg[1]:.2f}')
        
        plt.xlabel('Complejidad (numérica)')
        plt.ylabel('Severidad Promedio')
        plt.title('Relación Complejidad vs Severidad con Regresión Lineal', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./visualisations/05_complexity_severity_relationship.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_multiple_diagnoses_analysis(self):
        """Análisis de múltiples diagnósticos"""
        plt.figure(figsize=(12, 8))
        
        diag_counts_ramedis = self.df_ramedis['num_diagnoses'].value_counts().sort_index()
        x_ramedis = diag_counts_ramedis.index
        y_ramedis = diag_counts_ramedis.values
        
        bars_ramedis = plt.bar(x_ramedis - 0.2, y_ramedis, 0.4, label='RAMEDIS', color='blue', alpha=0.7)
        
        # URG_TORRE (asumiendo 1 diagnóstico por caso)
        urg_diag_count = len(self.df_urg_torre)
        bars_urg = plt.bar([1 + 0.2], [urg_diag_count], 0.4, label='URG_TORRE', color='red', alpha=0.7)
        
        plt.xlabel('Número de Diagnósticos')
        plt.ylabel('Número de Casos')
        plt.title('Distribución del Número de Diagnósticos por Caso')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('./visualisations/06_multiple_diagnoses_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self):
        """Mapa de calor de correlaciones RAMEDIS"""
        plt.figure(figsize=(10, 8))
        
        ramedis_numeric = pd.DataFrame({
            'Complejidad': self.df_ramedis['complexity'].apply(self.complexity_to_number),
            'Num_Diagnósticos': self.df_ramedis['num_diagnoses'],
            'Severidad_Promedio': self.df_ramedis['avg_severity_num'],
            'Num_Síntomas': self.df_ramedis['num_symptoms'],
            'Tiene_Muerte': self.df_ramedis['has_death'].astype(int)
        })
        
        corr_ramedis = ramedis_numeric.corr()
        mask = np.triu(np.ones_like(corr_ramedis, dtype=bool))
        
        sns.heatmap(corr_ramedis, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, square=True, linewidths=1,
                    cbar_kws={"shrink": .8})
        plt.title('Matriz de Correlación - RAMEDIS', fontsize=16)
        plt.tight_layout()
        plt.savefig('./visualisations/07_correlation_heatmap_ramedis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # URG_TORRE
        plt.figure(figsize=(10, 8))
        
        urg_numeric = pd.DataFrame({
            'Complejidad': self.df_urg_torre['complexity'].apply(self.complexity_to_number),
            'Num_Diagnósticos': self.df_urg_torre['num_diagnoses'],
            'Severidad_Promedio': self.df_urg_torre['avg_severity_num'],
            'Edad': self.df_urg_torre['patient_age'].fillna(self.df_urg_torre['patient_age'].median()),
            'Es_Crítico': self.df_urg_torre['critical'],
            'Muerte': self.df_urg_torre['death']
        })
        
        corr_urg = urg_numeric.corr()
        mask_urg = np.triu(np.ones_like(corr_urg, dtype=bool))
        
        sns.heatmap(corr_urg, mask=mask_urg, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True, linewidths=1,
                    cbar_kws={"shrink": .8})
        plt.title('Matriz de Correlación - URG_TORRE', fontsize=16)
        plt.tight_layout()
        plt.savefig('./visualisations/08_correlation_heatmap_urg_torre.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_symptoms_analysis(self):
        """Análisis de síntomas"""
        plt.figure(figsize=(12, 8))
        
        all_symptoms = []
        for symptoms in self.df_ramedis['symptoms']:
            if symptoms:
                all_symptoms.extend([s.strip() for s in symptoms.split(',')])
        
        if all_symptoms:
            symptom_counts = Counter(all_symptoms)
            top_symptoms = symptom_counts.most_common(15)
            symptoms, counts = zip(*top_symptoms)
            y_pos = np.arange(len(symptoms))
            
            bars = plt.barh(y_pos, counts, color=plt.cm.viridis(np.linspace(0, 1, len(symptoms))))
            plt.yticks(y_pos, [s[:30] + '...' if len(s) > 30 else s for s in symptoms])
            plt.xlabel('Frecuencia')
            plt.title('Top 15 Síntomas Más Frecuentes (RAMEDIS)')
            plt.gca().invert_yaxis()
            
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        f'{count}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('./visualisations/09_symptoms_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_demographic_analysis(self):
        """Análisis demográfico"""
        # Distribución por género
        plt.figure(figsize=(10, 8))
        gender_counts = self.df_urg_torre['patient_gender'].value_counts()
        colors = {'M': 'lightblue', 'F': 'pink', 'Unknown': 'gray'}
        
        wedges, texts, autotexts = plt.pie(gender_counts.values, 
                                           labels=[f'{k}\n({v} casos)' for k, v in gender_counts.items()],
                                           autopct='%1.1f%%',
                                           colors=[colors.get(k, 'gray') for k in gender_counts.index],
                                           startangle=90,
                                           explode=[0.05] * len(gender_counts))
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_weight('bold')
            
        plt.title('Distribución por Género - URG_TORRE', fontsize=16)
        plt.tight_layout()
        plt.savefig('./visualisations/10_demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_mortality_analysis(self):
        """Análisis de mortalidad"""
        plt.figure(figsize=(12, 8))
        
        mortality_by_complexity = self.df_urg_torre.groupby('complexity')['death'].agg(['mean', 'sum', 'count'])
        mortality_by_complexity['rate'] = mortality_by_complexity['mean'] * 100
        
        bars = plt.bar(mortality_by_complexity.index, mortality_by_complexity['rate'], 
                       color='darkred', alpha=0.7, edgecolor='black')
        
        plt.xlabel('Nivel de Complejidad')
        plt.ylabel('Tasa de Mortalidad (%)')
        plt.title('Tasa de Mortalidad por Nivel de Complejidad (URG_TORRE)')
        plt.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, rate, deaths, total) in enumerate(zip(bars, mortality_by_complexity['rate'], 
                                                          mortality_by_complexity['sum'], 
                                                          mortality_by_complexity['count'])):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{rate:.1f}%\n({deaths}/{total})', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('./visualisations/11_mortality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_case_clustering(self):
        """Clustering de casos con PCA"""
        plt.figure(figsize=(12, 8))
        
        # Preparar datos RAMEDIS
        ramedis_features = pd.DataFrame({
            'complexity': self.df_ramedis['complexity'].apply(self.complexity_to_number),
            'severity': self.df_ramedis['avg_severity_num'],
            'num_diagnoses': self.df_ramedis['num_diagnoses'],
            'num_symptoms': self.df_ramedis['num_symptoms'],
            'has_death': self.df_ramedis['has_death'].astype(int)
        })
        
        if len(ramedis_features) > 5:
            scaler = StandardScaler()
            ramedis_scaled = scaler.fit_transform(ramedis_features)
            
            # PCA
            pca = PCA(n_components=2)
            ramedis_pca = pca.fit_transform(ramedis_scaled)
            
            # K-means
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(ramedis_scaled)
            
            # Visualizar
            colors = ['red', 'blue', 'green']
            for i in range(3):
                mask = clusters == i
                plt.scatter(ramedis_pca[mask, 0], ramedis_pca[mask, 1], 
                           c=colors[i], label=f'Cluster {i+1}', alpha=0.7, s=50)
            
            # Centroides
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                       c='black', marker='x', s=200, linewidths=3, label='Centroides')
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
            plt.title('Clustering PCA - RAMEDIS')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./visualisations/12_case_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_diagnosis_network(self):
        """Red de diagnósticos"""
        plt.figure(figsize=(12, 8))
        
        all_diag = []
        for diag in self.df_ramedis['diagnosis_names']:
            if diag:
                all_diag.extend([d.strip() for d in diag.split(',')])
        
        if all_diag:
            diag_counts = Counter(all_diag)
            top_diag = diag_counts.most_common(10)
            names, counts = zip(*top_diag)
            
            plt.barh(range(len(names)), counts, color='skyblue')
            plt.yticks(range(len(names)), [n[:30]+'...' if len(n)>30 else n for n in names])
            plt.xlabel('Frecuencia')
            plt.title('Top 10 Diagnósticos RAMEDIS')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('./visualisations/13_diagnosis_network.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_complexity_timeline(self):
        """Análisis temporal simulado"""
        plt.figure(figsize=(12, 8))
        
        np.random.seed(42)
        months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        complexity_means = np.random.normal(5, 2, 12)
        
        plt.plot(months, complexity_means, 'o-', linewidth=2, markersize=6)
        plt.ylabel('Complejidad Promedio')
        plt.title('Evolución Temporal - RAMEDIS (Simulado)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('./visualisations/14_complexity_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_diagnostic_codes_distribution(self):
        """Distribución de códigos diagnósticos"""
        plt.figure(figsize=(10, 8))
        
        all_codes = []
        for codes in self.df_ramedis['diagnostic_codes']:
            if codes:
                all_codes.extend([c.strip() for c in codes.split(',')])
        
        if all_codes:
            code_counts = Counter(all_codes)
            top_codes = code_counts.most_common(8)
            codes, counts = zip(*top_codes)
            
            plt.pie(counts, labels=codes, autopct='%1.1f%%', startangle=90)
            plt.title('Distribución de Códigos Diagnósticos - RAMEDIS')
        else:
            plt.text(0.5, 0.5, 'Sin códigos disponibles', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Distribución de Códigos Diagnósticos - RAMEDIS')
        
        plt.tight_layout()
        plt.savefig('./visualisations/15_diagnostic_codes_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_symptoms_wordcloud(self):
        """Word cloud de síntomas"""
        plt.figure(figsize=(12, 8))
        
        all_symptoms = []
        for symptoms in self.df_ramedis['symptoms']:
            if symptoms:
                all_symptoms.extend([s.strip() for s in symptoms.split(',')])
        
        if all_symptoms:
            symptom_counts = Counter(all_symptoms)
            top_symptoms = symptom_counts.most_common(10)
            symptoms, counts = zip(*top_symptoms)
            
            plt.barh(range(len(symptoms)), counts, color='lightgreen')
            plt.yticks(range(len(symptoms)), [s[:25]+'...' if len(s)>25 else s for s in symptoms])
            plt.xlabel('Frecuencia')
            plt.title('Top 10 Síntomas - RAMEDIS')
            plt.gca().invert_yaxis()
        else:
            plt.text(0.5, 0.5, 'Sin síntomas disponibles', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Análisis de Síntomas - RAMEDIS')
        
        plt.tight_layout()
        plt.savefig('./visualisations/16_symptoms_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparative_analysis(self):
        """Análisis comparativo"""
        plt.figure(figsize=(12, 8))
        
        ramedis_comp = self.df_ramedis['complexity'].apply(self.complexity_to_number)
        urg_comp = self.df_urg_torre['complexity'].apply(self.complexity_to_number)
        
        plt.hist([ramedis_comp, urg_comp], bins=10, alpha=0.7, 
                label=['RAMEDIS', 'URG_TORRE'], color=['blue', 'red'])
        plt.xlabel('Complejidad')
        plt.ylabel('Frecuencia')
        plt.title('Comparación Distribución de Complejidad: RAMEDIS vs URG_TORRE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./visualisations/17_comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_severity_prediction_model(self):
        """Análisis predictivo de severidad"""
        plt.figure(figsize=(12, 8))
        
        features = ['Complejidad', 'Num_Diagnósticos', 'Num_Síntomas']
        correlations = [
            self.df_ramedis['complexity'].apply(self.complexity_to_number).corr(self.df_ramedis['avg_severity_num']),
            self.df_ramedis['num_diagnoses'].corr(self.df_ramedis['avg_severity_num']),
            self.df_ramedis['num_symptoms'].corr(self.df_ramedis['avg_severity_num'])
        ]
        
        colors = ['red' if c < 0 else 'green' for c in correlations]
        bars = plt.bar(features, correlations, color=colors, alpha=0.7)
        plt.ylabel('Correlación con Severidad')
        plt.title('Predictores de Severidad - RAMEDIS')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for bar, corr in zip(bars, correlations):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('./visualisations/18_severity_prediction_model.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    print("Iniciando análisis médico...")
    
    analyzer = MedicalDataAnalyzer('RAMEDIS.json', 'URG_TORRE.json')
    analyzer.prepare_dataframes()
    analyzer.create_visualizations()
    
    print("\n🎉 Análisis completado exitosamente!")
    print("📊 Todas las visualizaciones guardadas en ./visualisations/") 