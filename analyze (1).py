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

    # Strip Procore-appended total rows (no Number AND no Contract Company)
    proj = proj[~(proj['Number'].isna() & proj['Contract Company'].isna())].copy()

    # Clean cost columns — strip $, commas, convert to float
    for col in ['Original Contract Amount', 'Revised Contract Amount']:
        proj[col] = (
            proj[col].astype(str)
            .str.replace('[$,]', '', regex=True)
            .str.strip()
        )
        proj[col] = pd.to_numeric(proj[col], errors='coerce').fillna(0)

    # Exclude unassigned PO placeholders (Number starts with "PO")
    is_placeholder_po = proj['Number'].astype(str).str.startswith('PO')
    has_cost = (proj['Original Contract Amount'] > 0) | (proj['Revised Contract Amount'] > 0)

    # full_df: all real rows (for unique company counting — matches Excel SUMPRODUCT formula)
    full_df = proj[~is_placeholder_po].copy()

    # all_commitments: cost rows only (for financial totals)
    all_commitments = proj[~is_placeholder_po & has_cost].copy()

    # classifiable: cost rows with a vendor city (for local/nonlocal analysis)
    classifiable = all_commitments[all_commitments['Vendor City'].notna()].copy()

    return classifiable, all_commitments, full_df, cities


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

        # Non-FL vendors are always nonlocal (only if state column exists)
        if state is not None and pd.notna(state) and str(state).strip().upper() != 'FL':
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

    has_state = 'Vendor State' in subcontracts.columns
    results = subcontracts.apply(
        lambda row: resolve_city(
            row['Vendor City'],
            row['Vendor State'] if has_state else None
        ), axis=1
    )

    subcontracts = subcontracts.copy()
    subcontracts['County']   = [r[0] for r in results]
    subcontracts['Location Label'] = [r[1] for r in results]
    subcontracts['Is Local'] = [r[2] for r in results]

    return subcontracts


# ─────────────────────────────────────────────
# STEP 3: CALCULATE METRICS
# ─────────────────────────────────────────────

def calculate_metrics(subcontracts, all_commitments, full_df):
    metrics = {}

    # ── Total project cost (all commitments) ──
    metrics['total_original'] = all_commitments['Original Contract Amount'].sum()
    metrics['total_revised']  = all_commitments['Revised Contract Amount'].sum()
    metrics['total_variance'] = metrics['total_revised'] - metrics['total_original']
    metrics['total_variance_pct'] = (
        (metrics['total_variance'] / metrics['total_original'] * 100)
        if metrics['total_original'] > 0 else 0
    )

    # ── Subcontractor counts — unique company names across all rows
    # Matches Excel SUMPRODUCT(1/COUNTIF(...)) logic on the full table
    unique_subs   = full_df['Contract Company'].dropna().nunique()
    local_subs    = subcontracts[subcontracts['Is Local']]['Contract Company'].nunique()
    nonlocal_subs = subcontracts[~subcontracts['Is Local']]['Contract Company'].nunique()

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
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from datetime import date

    # ── Register Carlito (modern Calibri-compatible font) ──
    font_dir = '/usr/share/fonts/truetype/crosextra'
    pdfmetrics.registerFont(TTFont('Carlito',         f'{font_dir}/Carlito-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('Carlito-Bold',    f'{font_dir}/Carlito-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('Carlito-Italic',  f'{font_dir}/Carlito-Italic.ttf'))
    pdfmetrics.registerFontFamily('Carlito',
        normal='Carlito', bold='Carlito-Bold', italic='Carlito-Italic')

    F      = 'Carlito'
    F_BOLD = 'Carlito-Bold'

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.65*inch,
        leftMargin=0.65*inch,
        topMargin=0.65*inch,
        bottomMargin=0.65*inch
    )

    # ── Computed values (revised only) ──
    local_rev_pct    = (metrics['local_revised']    / metrics['subcontract_total_revised'] * 100) if metrics['subcontract_total_revised'] > 0 else 0
    nonlocal_rev_pct = (metrics['nonlocal_revised'] / metrics['subcontract_total_revised'] * 100) if metrics['subcontract_total_revised'] > 0 else 0
    local_var_pct    = metrics['local_variance_pct']
    nonlocal_var_pct = metrics['nonlocal_variance_pct']

    def currency(val):
        return f"${val:,.2f}"

    def signed_pct(val):
        return f"+{val:.1f}%" if val > 0 else f"{val:.1f}%"

    # ── Style helpers ──
    def h2_style(name):
        return ParagraphStyle(name, fontSize=9.5, fontName=F_BOLD,
                              textColor=colors.HexColor(BRAND_PRIMARY),
                              spaceBefore=12, spaceAfter=4,
                              letterSpacing=0.8)

    def label_style(name, align=TA_CENTER):
        return ParagraphStyle(name, fontSize=8, fontName=F,
                              textColor=colors.HexColor('#777777'),
                              alignment=align)

    def big_style(name, color=BRAND_PRIMARY):
        return ParagraphStyle(name, fontSize=26, fontName=F_BOLD,
                              textColor=colors.HexColor(color),
                              spaceAfter=1, alignment=TA_CENTER)

    elements = []

    # ═══════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════
    header_data = [[
        Paragraph(f"<font color='white'><b>LOCAL SUBCONTRACTOR ANALYSIS</b></font>",
                  ParagraphStyle('HT', fontSize=13, fontName=F_BOLD, textColor=colors.white)),
        Paragraph(f"<font color='white'>{metrics['project_name']}</font>",
                  ParagraphStyle('HP', fontSize=9, fontName=F, textColor=colors.white, alignment=TA_RIGHT))
    ]]
    header_table = Table(header_data, colWidths=[4*inch, 2.9*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor(BRAND_PRIMARY)),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',   (0,0), (0,0),   16),
        ('RIGHTPADDING',  (-1,0),(-1,0),  16),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 14))

    # ═══════════════════════════════════════════
    # HEADLINE CALLOUT — the number we want to shout
    # ═══════════════════════════════════════════
    headline_pct  = f"{local_rev_pct:.1f}%"
    headline_amt  = f"${metrics['local_revised']:,.0f}"
    headline_text = (
        f"of actual subcontract value — {headline_amt} of "
        f"${metrics['subcontract_total_revised']:,.0f} — was awarded to "
        f"<b>local subcontractors</b>."
    )

    callout_data = [[
        Paragraph(headline_pct,
                  ParagraphStyle('HLPct', fontSize=48, fontName=F_BOLD,
                                  textColor=colors.HexColor(BRAND_ACCENT),
                                  alignment=TA_CENTER, leading=52)),
        Paragraph(headline_text,
                  ParagraphStyle('HLText', fontSize=11, fontName=F,
                                  textColor=colors.HexColor(BRAND_PRIMARY),
                                  leading=16, spaceAfter=0))
    ]]
    callout_table = Table(callout_data, colWidths=[1.6*inch, 5.3*inch])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor(BRAND_LIGHT)),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',   (0,0), (0,0),   16),
        ('RIGHTPADDING',  (-1,0),(-1,0),  20),
        ('TOPPADDING',    (0,0), (-1,-1), 18),
        ('BOTTOMPADDING', (0,0), (-1,-1), 18),
        ('LINEBEFORE',    (0,0), (0,-1),  4, colors.HexColor(BRAND_ACCENT)),
    ]))
    elements.append(callout_table)
    elements.append(Spacer(1, 16))

    # ═══════════════════════════════════════════
    # SECTION 1 — PROJECT COST SUMMARY
    # ═══════════════════════════════════════════
    elements.append(Paragraph("PROJECT COST SUMMARY", h2_style('S1')))
    elements.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor(BRAND_ACCENT), spaceAfter=7))

    cost_data = [
        ['', 'Original Budget', 'Actual Cost', 'Variance', 'Variance %'],
        ['Total Project (All Commitments)',
         currency(metrics['total_original']),
         currency(metrics['total_revised']),
         currency(metrics['total_variance']),
         signed_pct(metrics['total_variance_pct'])],
    ]
    cost_table = Table(cost_data, colWidths=[2.3*inch, 1.2*inch, 1.2*inch, 1.2*inch, 0.9*inch])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor(BRAND_PRIMARY)),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
        ('FONTNAME',      (0,0), (-1,0),  F_BOLD),
        ('FONTNAME',      (0,1), (-1,-1), F),
        ('FONTNAME',      (0,1), (0,-1),  F_BOLD),
        ('FONTSIZE',      (0,0), (-1,-1), 8.5),
        ('ALIGN',         (1,0), (-1,-1), 'RIGHT'),
        ('BACKGROUND',    (0,1), (-1,-1), colors.HexColor(BRAND_LIGHT)),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#d8dde5')),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 9),
    ]))
    elements.append(cost_table)
    elements.append(Spacer(1, 14))

    # ═══════════════════════════════════════════
    # SECTION 2 — SUBCONTRACTOR SUMMARY (stat cards)
    # ═══════════════════════════════════════════
    elements.append(Paragraph("SUBCONTRACTOR SUMMARY", h2_style('S2')))
    elements.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor(BRAND_ACCENT), spaceAfter=7))

    stat_data = [
        [
            Paragraph(str(metrics['unique_subs']), big_style('N1')),
            Paragraph(str(metrics['local_subs']),  big_style('N2', BRAND_PRIMARY)),
            Paragraph(str(metrics['nonlocal_subs']), big_style('N3', '#888888')),
        ],
        [
            Paragraph("Total Companies", label_style('L1')),
            Paragraph("Local",           label_style('L2')),
            Paragraph("Nonlocal",        label_style('L3')),
        ],
    ]
    stat_table = Table(stat_data, colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
    stat_table.setStyle(TableStyle([
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor(BRAND_LIGHT)),
        ('TOPPADDING',    (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,0),  2),
        ('BOTTOMPADDING', (0,1), (-1,1),  12),
        ('LINEAFTER',     (0,0), (1,-1),  0.5, colors.HexColor('#c8cfd8')),
        ('LINEBEFORE',    (0,0), (0,-1),  3, colors.HexColor(BRAND_ACCENT)),
    ]))
    elements.append(stat_table)
    elements.append(Spacer(1, 14))

    # ═══════════════════════════════════════════
    # SECTION 3 — LOCAL VS NONLOCAL (revised only)
    # ═══════════════════════════════════════════
    elements.append(Paragraph("LOCAL VS. NONLOCAL PERFORMANCE  (Actual / Revised Amounts)", h2_style('S3')))
    elements.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor(BRAND_ACCENT), spaceAfter=7))

    perf_data = [
        ['', 'Original Budget', 'Actual Cost', 'Variance', 'Variance %', '% of Actual Spend'],
        ['Local',
         currency(metrics['local_original']),
         currency(metrics['local_revised']),
         currency(metrics['local_variance']),
         signed_pct(local_var_pct),
         f"{local_rev_pct:.1f}%"],
        ['Nonlocal',
         currency(metrics['nonlocal_original']),
         currency(metrics['nonlocal_revised']),
         currency(metrics['nonlocal_variance']),
         signed_pct(nonlocal_var_pct),
         f"{nonlocal_rev_pct:.1f}%"],
    ]
    perf_table = Table(perf_data, colWidths=[0.85*inch, 1.15*inch, 1.15*inch, 1.1*inch, 0.9*inch, 1.35*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),  (-1,0),  colors.HexColor(BRAND_PRIMARY)),
        ('TEXTCOLOR',     (0,0),  (-1,0),  colors.white),
        ('FONTNAME',      (0,0),  (-1,0),  F_BOLD),
        ('FONTNAME',      (0,1),  (-1,-1), F),
        ('FONTNAME',      (0,1),  (0,-1),  F_BOLD),
        ('FONTNAME',      (-1,1), (-1,-1), F_BOLD),   # % of Actual Spend col bold
        ('FONTSIZE',      (0,0),  (-1,-1), 8.5),
        ('ALIGN',         (1,0),  (-1,-1), 'RIGHT'),
        ('ROWBACKGROUNDS',(0,1),  (-1,-1), [colors.HexColor(BRAND_LIGHT), colors.white]),
        ('GRID',          (0,0),  (-1,-1), 0.4, colors.HexColor('#d8dde5')),
        ('TOPPADDING',    (0,0),  (-1,-1), 7),
        ('BOTTOMPADDING', (0,0),  (-1,-1), 7),
        ('LEFTPADDING',   (0,0),  (-1,-1), 9),
    ]))
    elements.append(perf_table)
    elements.append(Spacer(1, 14))

    # ═══════════════════════════════════════════
    # SECTION 4 — COUNTY BREAKDOWN
    # ═══════════════════════════════════════════
    elements.append(Paragraph("LOCAL SUBCONTRACT VALUE BY COUNTY  (Actual / Revised Amounts)", h2_style('S4')))
    elements.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor(BRAND_ACCENT), spaceAfter=7))
    elements.append(RLImage(chart_path, width=6.8*inch,
                             height=max(2.4*inch, len(metrics['county_breakdown']) * 0.48*inch + 0.8*inch)))
    elements.append(Spacer(1, 6))

    cb = metrics['county_breakdown']
    county_data = [['County', '# Contracts', 'Actual Cost', '% of Local Spend']]
    for _, row in cb.sort_values('Revised Amount', ascending=False).iterrows():
        county_data.append([
            row['County'],
            str(int(row['# Contracts'])),
            currency(row['Revised Amount']),
            f"{row['Share %']:.1f}%"
        ])
    county_data.append([
        f"Total Local ({len(cb)} {'county' if len(cb)==1 else 'counties'})",
        str(int(cb['# Contracts'].sum())),
        currency(cb['Revised Amount'].sum()),
        "100.0%"
    ])

    county_table = Table(county_data, colWidths=[1.9*inch, 1.0*inch, 1.9*inch, 1.8*inch])
    county_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),  (-1,0),  colors.HexColor(BRAND_PRIMARY)),
        ('TEXTCOLOR',     (0,0),  (-1,0),  colors.white),
        ('FONTNAME',      (0,0),  (-1,0),  F_BOLD),
        ('FONTNAME',      (0,1),  (-1,-1), F),
        ('FONTNAME',      (0,1),  (0,-1),  F_BOLD),
        ('FONTNAME',      (0,-1), (-1,-1), F_BOLD),
        ('FONTSIZE',      (0,0),  (-1,-1), 8.5),
        ('ALIGN',         (1,0),  (-1,-1), 'RIGHT'),
        ('ROWBACKGROUNDS',(0,1),  (-1,-2), [colors.HexColor(BRAND_LIGHT), colors.white]),
        ('BACKGROUND',    (0,-1), (-1,-1), colors.HexColor(BRAND_LIGHT)),
        ('TEXTCOLOR',     (0,-1), (-1,-1), colors.HexColor(BRAND_PRIMARY)),
        ('GRID',          (0,0),  (-1,-1), 0.4, colors.HexColor('#d8dde5')),
        ('TOPPADDING',    (0,0),  (-1,-1), 7),
        ('BOTTOMPADDING', (0,0),  (-1,-1), 7),
        ('LEFTPADDING',   (0,0),  (-1,-1), 9),
    ]))
    elements.append(Spacer(1, 6))
    elements.append(county_table)

    # ═══════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════
    elements.append(Spacer(1, 18))
    elements.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor('#cccccc'), spaceAfter=5))
    elements.append(Paragraph(
        f"Generated {date.today().strftime('%B %d, %Y')}  ·  Source: {os.path.basename(project_csv_path)}  ·  All figures reflect revised / actual contract amounts",
        ParagraphStyle('Footer', fontSize=7, fontName=F,
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
    subcontracts, all_commitments, full_df, cities_df = load_data(project_csv, cities_csv)

    print("Classifying locations...")
    subcontracts = classify_locations(subcontracts, cities_df)

    print("Calculating metrics...")
    metrics = calculate_metrics(subcontracts, all_commitments, full_df)

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
