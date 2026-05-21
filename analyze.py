"""
Local vs Nonlocal Subcontractor Analysis Tool
Analyzes Procore commitment exports against a local cities reference file.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import get_close_matches
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import os
import sys

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Brand colors (swap hex codes to match your company)
BRAND_PRIMARY   = "#1B3A6B"   # dark navy
BRAND_ACCENT    = "#C8960C"   # gold
BRAND_LIGHT     = "#E8EDF4"   # light blue-grey
NONLOCAL_COLOR  = "#B0B8C1"   # muted grey for nonlocal bars

FUZZY_CUTOFF = 0.75  # confidence threshold for fuzzy city matching

# ─────────────────────────────────────────────
# STEP 1: LOAD & CLEAN DATA
# ─────────────────────────────────────────────

def load_data(project_csv_path, cities_csv_path):
    proj = pd.read_csv(project_csv_path)
    cities = pd.read_csv(cities_csv_path)

    # Drop only blank template rows (no vendor AND no cost)
    # Purchase Orders with real vendors and costs are included in ALL analysis
    proj = proj[
        proj['Vendor'].notna() &
        ((proj['Original Contract Amount'] > 0) | (proj['Revised Contract Amount'] > 0))
    ].copy()

    # All commitments used for both total cost AND local/nonlocal analysis
    # (POs with real vendors count toward local participation)
    all_commitments = proj.copy()

    return all_commitments, all_commitments, cities


# ─────────────────────────────────────────────
# STEP 2: LOCAL CLASSIFICATION
# ─────────────────────────────────────────────

def classify_locations(subcontracts, cities_df):
    # Build lookup: lowercase city name -> county
    city_to_county = dict(zip(
        cities_df['City'].str.strip().str.lower(),
        cities_df['County']
    ))
    canonical_cities = list(city_to_county.keys())

    def resolve_city(city, state):
        if pd.isna(city) or str(city).strip() == '':
            return None, 'Unknown', False

        # Non-FL vendors are always nonlocal
        if pd.notna(state) and str(state).strip().upper() != 'FL':
            return None, 'Out of State', False

        key = str(city).strip().lower()

        # Direct match
        if key in city_to_county:
            return city_to_county[key], city_to_county[key], True

        # Fuzzy match
        matches = get_close_matches(key, canonical_cities, n=1, cutoff=FUZZY_CUTOFF)
        if matches:
            matched_county = city_to_county[matches[0]]
            return matched_county, matched_county, True

        return None, 'Other FL', False

    results = subcontracts.apply(
        lambda row: resolve_city(row['Vendor City'], row['Vendor State']), axis=1
    )

    subcontracts = subcontracts.copy()
    subcontracts['County']   = [r[0] for r in results]
    subcontracts['Location Label'] = [r[1] for r in results]
    subcontracts['Is Local'] = [r[2] for r in results]

    return subcontracts


# ─────────────────────────────────────────────
# STEP 3: CALCULATE METRICS
# ─────────────────────────────────────────────

def calculate_metrics(subcontracts, all_commitments):
    metrics = {}

    # ── Total project cost (all commitments) ──
    metrics['total_original'] = all_commitments['Original Contract Amount'].sum()
    metrics['total_revised']  = all_commitments['Revised Contract Amount'].sum()
    metrics['total_variance'] = metrics['total_revised'] - metrics['total_original']
    metrics['total_variance_pct'] = (
        (metrics['total_variance'] / metrics['total_original'] * 100)
        if metrics['total_original'] > 0 else 0
    )

    # ── Subcontractor counts (all vendors incl POs) ──
    unique_subs   = subcontracts['Contract Company'].nunique()
    local_subs    = subcontracts[subcontracts['Is Local']]['Contract Company'].nunique()
    nonlocal_subs = unique_subs - local_subs

    metrics['unique_subs']   = unique_subs
    metrics['local_subs']    = local_subs
    metrics['nonlocal_subs'] = nonlocal_subs

    # ── Monetary share ──
    total_sub_original = subcontracts['Original Contract Amount'].sum()
    total_sub_revised  = subcontracts['Revised Contract Amount'].sum()

    local_df    = subcontracts[subcontracts['Is Local']]
    nonlocal_df = subcontracts[~subcontracts['Is Local']]

    local_orig    = local_df['Original Contract Amount'].sum()
    local_rev     = local_df['Revised Contract Amount'].sum()
    nonlocal_orig = nonlocal_df['Original Contract Amount'].sum()
    nonlocal_rev  = nonlocal_df['Revised Contract Amount'].sum()

    metrics['subcontract_total_original'] = total_sub_original
    metrics['subcontract_total_revised']  = total_sub_revised

    metrics['local_original']    = local_orig
    metrics['local_revised']     = local_rev
    metrics['nonlocal_original'] = nonlocal_orig
    metrics['nonlocal_revised']  = nonlocal_rev

    metrics['local_share_pct']    = (local_orig / total_sub_original * 100) if total_sub_original > 0 else 0
    metrics['nonlocal_share_pct'] = (nonlocal_orig / total_sub_original * 100) if total_sub_original > 0 else 0

    # ── Budget vs actual variance by local/nonlocal ──
    metrics['local_variance']    = local_rev - local_orig
    metrics['nonlocal_variance'] = nonlocal_rev - nonlocal_orig

    metrics['local_variance_pct'] = (
        (metrics['local_variance'] / local_orig * 100) if local_orig > 0 else 0
    )
    metrics['nonlocal_variance_pct'] = (
        (metrics['nonlocal_variance'] / nonlocal_orig * 100) if nonlocal_orig > 0 else 0
    )

    # ── County breakdown (local only) ──
    county_group = local_df.groupby('County').agg(
        Original_Amount=('Original Contract Amount', 'sum'),
        Revised_Amount=('Revised Contract Amount', 'sum'),
        Num_Contracts=('Contract Company', 'count')
    ).reset_index()
    county_group.columns = ['County', 'Original Amount', 'Revised Amount', '# Contracts']
    county_group = county_group.sort_values('Revised Amount', ascending=True)
    county_group['Share %'] = county_group['Revised Amount'] / total_sub_revised * 100

    metrics['county_breakdown'] = county_group

    # ── Project name ──
    metrics['project_name'] = subcontracts['Project Name'].iloc[0] if len(subcontracts) > 0 else 'Project'

    return metrics


# ─────────────────────────────────────────────
# STEP 4: GENERATE CHART
# ─────────────────────────────────────────────

def generate_county_chart(county_breakdown, output_path):
    fig, ax = plt.subplots(figsize=(7, max(3, len(county_breakdown) * 0.6 + 1)))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars = ax.barh(
        county_breakdown['County'],
        county_breakdown['Revised Amount'],
        color=BRAND_PRIMARY,
        height=0.55,
        zorder=3
    )

    for bar, val, pct in zip(bars, county_breakdown['Revised Amount'], county_breakdown['Share %']):
        ax.text(
            bar.get_width() + (county_breakdown['Revised Amount'].max() * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}  ({pct:.1f}%)",
            va='center', ha='left',
            fontsize=9, color='#333333'
        )

    ax.set_xlabel('Revised Contract Amount ($)', fontsize=9, color='#555555')
    ax.set_title('Local Subcontract Value by County (Revised)', fontsize=11,
                 fontweight='bold', color=BRAND_PRIMARY, pad=10)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.tick_params(axis='both', labelsize=9, colors='#555555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    ax.set_xlim(0, county_breakdown['Revised Amount'].max() * 1.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ─────────────────────────────────────────────
# STEP 5: BUILD PDF REPORT
# ─────────────────────────────────────────────

def build_pdf(metrics, chart_path, output_path, project_csv_path):

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    # ── Styles ──
    styles = getSampleStyleSheet()

    style_h1 = ParagraphStyle('H1', fontSize=18, fontName='Helvetica-Bold',
                               textColor=colors.HexColor(BRAND_PRIMARY),
                               spaceAfter=4)
    style_h2 = ParagraphStyle('H2', fontSize=12, fontName='Helvetica-Bold',
                               textColor=colors.HexColor(BRAND_PRIMARY),
                               spaceBefore=14, spaceAfter=6)
    style_sub = ParagraphStyle('Sub', fontSize=9, fontName='Helvetica',
                                textColor=colors.HexColor('#666666'), spaceAfter=12)
    style_body = ParagraphStyle('Body', fontSize=9, fontName='Helvetica',
                                 textColor=colors.HexColor('#333333'), spaceAfter=6)
    style_label = ParagraphStyle('Label', fontSize=8, fontName='Helvetica',
                                  textColor=colors.HexColor('#888888'))
    style_big_num = ParagraphStyle('BigNum', fontSize=22, fontName='Helvetica-Bold',
                                    textColor=colors.HexColor(BRAND_PRIMARY), spaceAfter=2)
    style_caption = ParagraphStyle('Caption', fontSize=8, fontName='Helvetica',
                                    textColor=colors.HexColor('#888888'), alignment=TA_CENTER)

    def currency(val):
        return f"${val:,.2f}"

    def pct(val):
        sign = '+' if val > 0 else ''
        return f"{sign}{val:.1f}%"

    elements = []

    # ── Header bar ──
    header_data = [[
        Paragraph(f"<font color='white'><b>LOCAL SUBCONTRACTOR ANALYSIS</b></font>", 
                  ParagraphStyle('HdrTitle', fontSize=14, fontName='Helvetica-Bold',
                                  textColor=colors.white)),
        Paragraph(f"<font color='white'>{metrics['project_name']}</font>",
                  ParagraphStyle('HdrProj', fontSize=9, fontName='Helvetica',
                                  textColor=colors.white, alignment=TA_RIGHT))
    ]]
    header_table = Table(header_data, colWidths=[4*inch, 2.75*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(BRAND_PRIMARY)),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (0,0), 14),
        ('RIGHTPADDING', (-1,0), (-1,0), 14),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('ROUNDEDCORNERS', [4,4,4,4]),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 16))

    # ── Section 1: Project Cost Summary ──
    elements.append(Paragraph("PROJECT COST SUMMARY", style_h2))
    elements.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor(BRAND_ACCENT), spaceAfter=8))

    cost_data = [
        ['', 'Original Budget', 'Actual Cost', 'Variance', 'Variance %'],
        ['Total Project (All Commitments)',
         currency(metrics['total_original']),
         currency(metrics['total_revised']),
         currency(metrics['total_variance']),
         pct(metrics['total_variance_pct'])],
    ]

    cost_table = Table(cost_data, colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 0.9*inch])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(BRAND_PRIMARY)),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor(BRAND_LIGHT)),
        ('BACKGROUND', (0,2), (-1,2), colors.white),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor(BRAND_LIGHT), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(cost_table)
    elements.append(Spacer(1, 16))

    # ── Section 2: Subcontractor Summary ──
    elements.append(Paragraph("SUBCONTRACTOR SUMMARY", style_h2))
    elements.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor(BRAND_ACCENT), spaceAfter=8))

    # Stat boxes
    stat_data = [[
        Paragraph(f"<b>{metrics['unique_subs']}</b>", style_big_num),
        Paragraph(f"<b>{metrics['local_subs']}</b>", style_big_num),
        Paragraph(f"<b>{metrics['nonlocal_subs']}</b>", style_big_num),
    ],[
        Paragraph("Total Unique Subcontractors", style_label),
        Paragraph("Local Subcontractors", style_label),
        Paragraph("Nonlocal Subcontractors", style_label),
    ]]
    stat_table = Table(stat_data, colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
    stat_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(BRAND_LIGHT)),
        ('ROUNDEDCORNERS', [4,4,4,4]),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 2),
        ('BOTTOMPADDING', (0,1), (-1,1), 10),
        ('LINEAFTER', (0,0), (1,-1), 0.5, colors.HexColor('#cccccc')),
    ]))
    elements.append(stat_table)
    elements.append(Spacer(1, 16))

    # ── Section 3: Local vs Nonlocal Performance ──
    elements.append(Paragraph("LOCAL VS. NONLOCAL PERFORMANCE", style_h2))
    elements.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor(BRAND_ACCENT), spaceAfter=8))

    perf_data = [
        ['', 'Original Budget', 'Actual Cost', 'Variance', 'Variance %', '% of Sub Budget'],
        ['Local',
         currency(metrics['local_original']),
         currency(metrics['local_revised']),
         currency(metrics['local_variance']),
         pct(metrics['local_variance_pct']),
         f"{metrics['local_share_pct']:.1f}%"],
        ['Nonlocal',
         currency(metrics['nonlocal_original']),
         currency(metrics['nonlocal_revised']),
         currency(metrics['nonlocal_variance']),
         pct(metrics['nonlocal_variance_pct']),
         f"{metrics['nonlocal_share_pct']:.1f}%"],
    ]

    perf_table = Table(perf_data, colWidths=[0.9*inch, 1.15*inch, 1.15*inch, 1.15*inch, 0.9*inch, 1.25*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(BRAND_PRIMARY)),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor(BRAND_LIGHT), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(perf_table)
    elements.append(Spacer(1, 16))

    # ── Section 4: County Chart ──
    elements.append(Paragraph("LOCAL SUBCONTRACT VALUE BY COUNTY", style_h2))
    elements.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor(BRAND_ACCENT), spaceAfter=8))
    elements.append(RLImage(chart_path, width=6.5*inch, height=max(2.5*inch, len(metrics['county_breakdown']) * 0.45*inch + 0.8*inch)))
    elements.append(Spacer(1, 6))

    # County data table
    county_data = [['County', '# Contracts', 'Original Amount', 'Revised Amount', '% of Local']]
    for _, row in metrics['county_breakdown'].sort_values('Revised Amount', ascending=False).iterrows():
        county_data.append([
            row['County'],
            str(int(row['# Contracts'])),
            currency(row['Original Amount']),
            currency(row['Revised Amount']),
            f"{row['Share %']:.1f}%"
        ])
    # Totals row
    cb = metrics['county_breakdown']
    county_data.append([
        f"Total Local ({len(cb)} counties)",
        str(int(cb['# Contracts'].sum())),
        currency(cb['Original Amount'].sum()),
        currency(cb['Revised Amount'].sum()),
        "100.0%"
    ])

    county_table = Table(county_data, colWidths=[1.5*inch, 0.8*inch, 1.4*inch, 1.4*inch, 0.85*inch])
    county_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(BRAND_PRIMARY)),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor(BRAND_ACCENT + '33')),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor(BRAND_LIGHT), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(Spacer(1, 8))
    elements.append(county_table)

    # ── Footer ──
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor('#cccccc'), spaceAfter=6))
    from datetime import date
    elements.append(Paragraph(
        f"Generated {date.today().strftime('%B %d, %Y')}  ·  Source: {os.path.basename(project_csv_path)}",
        ParagraphStyle('Footer', fontSize=7, fontName='Helvetica',
                        textColor=colors.HexColor('#aaaaaa'), alignment=TA_CENTER)
    ))

    doc.build(elements)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(project_csv, cities_csv, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, 'county_chart.png')
    pdf_path   = os.path.join(output_dir, 'local_analysis_report.pdf')

    print("Loading data...")
    subcontracts, all_commitments, cities_df = load_data(project_csv, cities_csv)

    print("Classifying locations...")
    subcontracts = classify_locations(subcontracts, cities_df)

    print("Calculating metrics...")
    metrics = calculate_metrics(subcontracts, all_commitments)

    print("Generating chart...")
    generate_county_chart(metrics['county_breakdown'], chart_path)

    print("Building PDF report...")
    build_pdf(metrics, chart_path, pdf_path, project_csv)

    print(f"\n✓ Report saved to: {pdf_path}")
    return metrics, pdf_path


if __name__ == "__main__":
    project_csv = sys.argv[1] if len(sys.argv) > 1 else "data/project_data.csv"
    cities_csv  = sys.argv[2] if len(sys.argv) > 2 else "data/local_cities.csv"
    output_dir  = sys.argv[3] if len(sys.argv) > 3 else "output"
    run(project_csv, cities_csv, output_dir)
