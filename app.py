"""
Who outlives whom: a 28-year race between patients and their pacemaker generators
Analysis script — BMJ Christmas 2025 submission (v20)

Author:      Juan Pablo Guzmán, MD
Institution: Arrhythmia and Cardiac Pacing Section,
             Hospital Churruca-Visca, Buenos Aires, Argentina
Contact:     jpguzman1988@gmail.com

AI disclosure: An AI language model (Claude, Anthropic) was used for
               English-language editing and stylistic formatting only.
               All analyses and intellectual content were performed by the authors.

v20 changes vs v19:
  1. Cohort restricted to permanent cardiac pacemakers (PM) only. Implantable
     cardioverter-defibrillators, cardiac resynchronisation therapy devices,
     and implantable loop recorders are handled by a pre-specified filter on
     the institutional registry (see classify_pm()).
  2. Generator endpoint analysed with Aalen-Johansen cumulative incidence
     function, treating death as a competing event. Kaplan-Meier is retained
     only for the patient endpoint.
  3. Age stratification now excludes patients with missing date of birth
     (n = 69 in the PM analytic cohort).
  4. STROBE flow reflects the exclusion of non-pacemaker devices as a
     pre-analysis step.

Requirements:
    pandas      >= 2.0
    numpy       >= 1.24
    lifelines   >= 0.29
    matplotlib  >= 3.7
    openpyxl    >= 3.1

Usage:
    python analysis_bmj_christmas_v20.py

Input:
    base_marcapasos_v4.xlsx   (sheet: "Base Marcapasos")
        Prepared by prior step: the institutional pacing registry restricted
        to permanent pacemakers. See classify_pm() logic in the preparation
        script for classification rules.

        Required date columns:
            - Fecha de Implante
            - Fecha de Recambio       (NaN if no elective replacement)
            - ultimo_control          (date of last clinical contact)
            - Fecha de Óbito          (date of death for deceased patients)
            - Fecha de Nacimiento
        Required flag column:
            - Óbito                   (boolean: deceased yes/no)
        Required categorical column:
            - Sexo                    ('Masculino' or 'Femenino')
        Required count column:
            - n_controles             (total follow-up visits per patient)

Outputs:
    figure1_km_patient_vs_generator.png
    figure2_paradox_frequent_visitor.png
    figure3_race_outcomes.png
    figureS1_strobe_flow.png
    figureS2_age_stratified.png
    figureS3_sex_stratified.png
    results_summary_v20.txt

Key methodological choices:
  - Patient event time:  date of death when available; last clinical contact
                         used for deceased patients without a confirmed death
                         date (conservative approximation).
  - Patient censoring:   min(ultimo_control, CUTOFF_DATE), whichever first
                         (conservative — avoids assuming survival between
                         last contact and registry closure).
  - Generator endpoint:  Aalen-Johansen estimator of the cumulative incidence
                         of elective replacement, with death from any cause as
                         a competing event. Freedom-from-replacement = 1 - CIF.
  - Age stratification:  patients with missing date of birth (n = 69) are
                         excluded from the <75 vs >=75 analyses only.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, AalenJohansenFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE  = "base_marcapasos_v4.xlsx"
SHEET_NAME  = "Base Marcapasos"
CUTOFF_DATE = pd.Timestamp("2025-10-01")

COL_PAC = '#C0392B'
COL_GEN = '#2471A3'
QTILES  = ['Q1: 1–4', 'Q2: 5–8', 'Q3: 9–17', 'Q4: ≥18']
QCOLORS = ['#E74C3C', '#E67E22', '#27AE60', '#2471A3']


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING AND PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

base = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

for col in ['Fecha de Implante', 'Fecha de Recambio', 'ultimo_control',
            'Fecha de Óbito', 'Fecha de Nacimiento']:
    base[col] = pd.to_datetime(base[col], errors='coerce')

base['obito']    = base['Óbito'].astype(bool)
base['recambio'] = base['Fecha de Recambio'].notna()
base['edad']     = (base['Fecha de Implante']
                    - base['Fecha de Nacimiento']).dt.days / 365.25


# --- Patient event time (t_pac) ---
def _t_pac(row):
    if pd.isna(row['Fecha de Implante']):
        return np.nan
    if row['obito']:
        end = row['Fecha de Óbito'] if pd.notna(row['Fecha de Óbito']) \
              else row['ultimo_control']
        if pd.isna(end):
            return np.nan
    else:
        if pd.notna(row['ultimo_control']):
            end = min(row['ultimo_control'], CUTOFF_DATE)
        else:
            end = CUTOFF_DATE
    return (end - row['Fecha de Implante']).days / 365.25


base['t_pac'] = base.apply(_t_pac, axis=1)


# --- Competing-risks time (t_cr) and event code (e_cr) ---
# e_cr = 0 -> censored
#        1 -> elective replacement
#        2 -> death (competing event)
def _cr(row):
    if pd.isna(row['Fecha de Implante']):
        return pd.Series({'t_cr': np.nan, 'e_cr': np.nan})
    candidates = []
    if row['recambio']:
        candidates.append(('rep', row['Fecha de Recambio']))
    if row['obito']:
        dd = row['Fecha de Óbito'] if pd.notna(row['Fecha de Óbito']) \
             else row['ultimo_control']
        if pd.notna(dd):
            candidates.append(('death', dd))
    if not row['obito']:
        c = min(row['ultimo_control'], CUTOFF_DATE) \
            if pd.notna(row['ultimo_control']) else CUTOFF_DATE
        candidates.append(('cens', c))
    if not candidates:
        return pd.Series({'t_cr': np.nan, 'e_cr': np.nan})
    candidates.sort(key=lambda x: x[1])
    ev, d = candidates[0]
    t = (d - row['Fecha de Implante']).days / 365.25
    if t <= 0:
        return pd.Series({'t_cr': np.nan, 'e_cr': np.nan})
    code = {'cens': 0, 'rep': 1, 'death': 2}[ev]
    return pd.Series({'t_cr': t, 'e_cr': code})


base[['t_cr', 'e_cr']] = base.apply(_cr, axis=1)

# Analytic cohort: valid patient follow-up and valid CR time
df = base[(base['t_pac'] > 0) & (base['t_cr'] > 0)].copy()

# Visit quartiles (equal-count groupings)
df['q_ctrl'] = pd.qcut(df['n_controles'].dropna(), q=4, labels=QTILES)

# Age strata — exclude missing DOB from this variable
df['age_strat'] = np.select(
    [df['edad'] < 75, df['edad'] >= 75],
    ['<75 yr', '≥75 yr'],
    default=None
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRIMARY ANALYSIS — PATIENT SURVIVAL (KM) vs GENERATOR CIF (AJ)
# ─────────────────────────────────────────────────────────────────────────────

# Patient KM (standard)
kmf_pac = KaplanMeierFitter().fit(df['t_pac'], event_observed=df['obito'])


def median_ci(kmf):
    ci = kmf.confidence_interval_
    t  = ci.index.values
    lo = ci.iloc[:, 0].values
    hi = ci.iloc[:, 1].values
    idx_lo = np.where(lo <= 0.5)[0]
    idx_hi = np.where(hi <= 0.5)[0]
    ci_lo = t[idx_lo[0]] if len(idx_lo) else np.inf
    ci_hi = t[idx_hi[0]] if len(idx_hi) else np.inf
    return ci_lo, ci_hi


med_pac = kmf_pac.median_survival_time_
ci_lo_pac, ci_hi_pac = median_ci(kmf_pac)

# Generator Aalen-Johansen (death as competing event)
aj_gen = AalenJohansenFitter()
aj_gen.fit(df['t_cr'], df['e_cr'], event_of_interest=1)

# Median generator survival from AJ (time at which freedom = 0.5)
# CIF-based: median generator replacement time is when CIF >= 0.5
cif_series = aj_gen.cumulative_density_.iloc[:, 0]
idx_50 = np.where(cif_series.values >= 0.5)[0]
med_gen_aj = cif_series.index[idx_50[0]] if len(idx_50) else np.inf


def aj_freedom(aj, t):
    """Return freedom-from-event = 1 - CIF(t) for an AJ fitter."""
    return float(1 - aj.predict(t))


# Outcome cross-classification
n          = len(df)
died_orig  = (df['obito'] & ~df['recambio']).sum()
died_recam = (df['obito'] &  df['recambio']).sum()
vivo_orig  = (~df['obito'] & ~df['recambio']).sum()
vivo_recam = (~df['obito'] &  df['recambio']).sum()

# Survival / freedom at timepoints
timepoints = [1, 3, 5, 7, 9, 10, 12, 15, 18, 20]
surv_table = pd.DataFrame({
    't':             timepoints,
    'S_patient':     [float(kmf_pac.survival_function_at_times(t).values[0])
                      for t in timepoints],
    'Freedom_gen':   [aj_freedom(aj_gen, t) for t in timepoints],
})
surv_table['delta_pp'] = (surv_table['Freedom_gen']
                          - surv_table['S_patient']) * 100


# ─────────────────────────────────────────────────────────────────────────────
# 4. SECONDARY ANALYSIS — VISIT FREQUENCY AND SURVIVAL
# ─────────────────────────────────────────────────────────────────────────────

visit_stats = []
kmfs_q = {}
for q in QTILES:
    sub = df[df['q_ctrl'] == q].dropna(subset=['t_pac', 'obito'])
    kmf = KaplanMeierFitter().fit(sub['t_pac'], event_observed=sub['obito'])
    rate = sub['obito'].sum() / sub['t_pac'].sum() * 100
    visit_stats.append({
        'quartile':       q,
        'n':              len(sub),
        'deaths':         sub['obito'].sum(),
        'crude_mort_pct': sub['obito'].mean() * 100,
        'fu_median_yr':   sub['t_pac'].median(),
        'pt_years':       sub['t_pac'].sum(),
        'rate_per_100py': rate,
    })
    kmfs_q[q] = kmf

visit_df = pd.DataFrame(visit_stats)

total_visits = int(df['n_controles'].sum())
km_equator   = total_visits * 20 / 40075
hours_transit_years = total_visits * 2 / (24 * 365.25)


# ─────────────────────────────────────────────────────────────────────────────
# 5. AGE-STRATIFIED SUPPLEMENTARY ANALYSIS (<75 vs ≥75)
#    Restricted to patients with valid date of birth.
# ─────────────────────────────────────────────────────────────────────────────

df_age = df[df['age_strat'].notna()].copy()
n_age = len(df_age)
n_age_missing = len(df) - n_age

age_results = {}
for stratum in ['<75 yr', '≥75 yr']:
    sub = df_age[df_age['age_strat'] == stratum]
    kmf_p = KaplanMeierFitter().fit(sub['t_pac'], event_observed=sub['obito'])
    aj_g = AalenJohansenFitter()
    aj_g.fit(sub['t_cr'], sub['e_cr'], event_of_interest=1)
    deaths_no_rep = (sub['obito'] & ~sub['recambio']).sum()
    cif_series_s = aj_g.cumulative_density_.iloc[:, 0]
    idx_50_s = np.where(cif_series_s.values >= 0.5)[0]
    med_gen_s = cif_series_s.index[idx_50_s[0]] if len(idx_50_s) else np.inf
    age_results[stratum] = {
        'n':                len(sub),
        'deaths':           int(sub['obito'].sum()),
        'replacements':     int(sub['recambio'].sum()),
        'median_patient':   kmf_p.median_survival_time_,
        'median_generator': med_gen_s,
        'deaths_no_rep':    int(deaths_no_rep),
        'pct_deaths_no_rep': (deaths_no_rep / sub['obito'].sum() * 100
                              if sub['obito'].sum() > 0 else 0),
        'S_pac_9':          float(kmf_p.survival_function_at_times(9).values[0]),
        'freedom_gen_9':    aj_freedom(aj_g, 9),
        'kmf_pac':          kmf_p,
        'aj_gen':           aj_g,
    }

young = df_age[df_age['age_strat'] == '<75 yr']
old   = df_age[df_age['age_strat'] == '≥75 yr']
lr_age = logrank_test(young['t_pac'], old['t_pac'],
                      event_observed_A=young['obito'],
                      event_observed_B=old['obito'])


# ─────────────────────────────────────────────────────────────────────────────
# 5b. SEX-STRATIFIED SUPPLEMENTARY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

sex_results = {}
for label_es, label_en in [('Masculino', 'Men'), ('Femenino', 'Women')]:
    sub = df[df['Sexo'] == label_es]
    kmf_p = KaplanMeierFitter().fit(sub['t_pac'], event_observed=sub['obito'])
    aj_g = AalenJohansenFitter()
    aj_g.fit(sub['t_cr'], sub['e_cr'], event_of_interest=1)
    lo_s, hi_s = median_ci(kmf_p)
    deaths_no_rep = (sub['obito'] & ~sub['recambio']).sum()
    cif_series_s = aj_g.cumulative_density_.iloc[:, 0]
    idx_50_s = np.where(cif_series_s.values >= 0.5)[0]
    med_gen_s = cif_series_s.index[idx_50_s[0]] if len(idx_50_s) else np.inf
    sex_results[label_en] = {
        'label_es':         label_es,
        'n':                len(sub),
        'deaths':           int(sub['obito'].sum()),
        'replacements':     int(sub['recambio'].sum()),
        'median_patient':   kmf_p.median_survival_time_,
        'median_patient_ci': (lo_s, hi_s),
        'median_generator': med_gen_s,
        'deaths_no_rep':    int(deaths_no_rep),
        'pct_deaths_no_rep': (deaths_no_rep / sub['obito'].sum() * 100
                              if sub['obito'].sum() > 0 else 0),
        'S_pac_9':          float(kmf_p.survival_function_at_times(9).values[0]),
        'freedom_gen_9':    aj_freedom(aj_g, 9),
        'kmf_pac':          kmf_p,
        'aj_gen':           aj_g,
    }

male   = df[df['Sexo'] == 'Masculino']
female = df[df['Sexo'] == 'Femenino']
lr_sex_pac = logrank_test(male['t_pac'], female['t_pac'],
                          event_observed_A=male['obito'],
                          event_observed_B=female['obito'])
lr_sex_gen = logrank_test(male['t_cr'], female['t_cr'],
                          event_observed_A=(male['e_cr']==1),
                          event_observed_B=(female['e_cr']==1))


# ─────────────────────────────────────────────────────────────────────────────
# 6. STROBE FLOW COUNTS
# ─────────────────────────────────────────────────────────────────────────────

n_registry_pm = len(base)                        # PM-only registry (1321)
n_excluded    = n_registry_pm - n                # 33
invalid = base[~((base['t_pac'] > 0) & (base['t_cr'] > 0))]
n_excl_no_uc = (invalid['obito']
                & invalid['ultimo_control'].isna()
                & invalid['Fecha de Óbito'].isna()).sum()
n_excl_other = n_excluded - n_excl_no_uc


# ─────────────────────────────────────────────────────────────────────────────
# 7. RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

med_gen_str = ('Not reached' if np.isinf(med_gen_aj)
               else f'{med_gen_aj:.2f} yr')

summary = f"""
============================================================
RESULTS SUMMARY — Who outlives whom?  (v20, pacemakers only)
============================================================

COHORT (STROBE flow within pacemaker registry)
  PM registry:                 {n_registry_pm:,}
  Excluded:                    {n_excluded}
    - deceased without any
      follow-up date:          {n_excl_no_uc}
    - other invalid FU time:   {n_excl_other}
  Analytic PM cohort:          {n:,}

BASELINE
  Deaths:                      {df['obito'].sum():,} ({df['obito'].mean()*100:.1f}%)
  Generator replacements:      {df['recambio'].sum():,} ({df['recambio'].mean()*100:.1f}%)
  Total patient-years:         {df['t_pac'].sum():.0f}
  Median follow-up:            {df['t_pac'].median():.1f} yr  (IQR {df['t_pac'].quantile(.25):.1f}–{df['t_pac'].quantile(.75):.1f})
  FU range:                    {df['t_pac'].min()*12:.1f} months to {df['t_pac'].max():.1f} years
  Median age at implant:       {df['edad'].median():.1f} yr   (IQR {df['edad'].quantile(.25):.1f}–{df['edad'].quantile(.75):.1f})  [n={df['edad'].notna().sum()}]
  Missing DOB:                 {df['edad'].isna().sum()}
  Implant year range:          {df['Fecha de Implante'].min().year}–{df['Fecha de Implante'].max().year}

SEX DISTRIBUTION
  Men:                         {(df['Sexo']=='Masculino').sum()} ({(df['Sexo']=='Masculino').mean()*100:.1f}%)
  Women:                       {(df['Sexo']=='Femenino').sum()} ({(df['Sexo']=='Femenino').mean()*100:.1f}%)
  Median age men:              {df[df['Sexo']=='Masculino']['edad'].median():.1f} yr
  Median age women:            {df[df['Sexo']=='Femenino']['edad'].median():.1f} yr

PRIMARY SURVIVAL (patient KM, generator Aalen-Johansen)
  Median patient survival:     {med_pac:.2f} yr  (95% CI {ci_lo_pac:.2f}–{ci_hi_pac:.2f})
  Median generator (CIF=0.5):  {med_gen_str}

SURVIVAL AT TIMEPOINTS
  (Freedom_gen = 1 - Aalen-Johansen CIF of replacement)
{surv_table.to_string(index=False, float_format=lambda x: f'{x:.3f}')}

OUTCOME CROSS-CLASSIFICATION
  Died without elective replacement:    {died_orig:4d} ({died_orig/n*100:.1f}%)
  Alive without elective replacement:   {vivo_orig:4d} ({vivo_orig/n*100:.1f}%)
  Died after elective replacement:      {died_recam:4d} ({died_recam/n*100:.1f}%)
  Alive after elective replacement:     {vivo_recam:4d} ({vivo_recam/n*100:.1f}%)
  Died without replacement / all deaths: {died_orig}/{df['obito'].sum()} = {died_orig/df['obito'].sum()*100:.1f}%

VISIT FREQUENCY (secondary)
{visit_df.to_string(index=False, float_format=lambda x: f'{x:.2f}')}

  Total follow-up visits:      {total_visits:,}
  Transit time (2 h/visit):    {hours_transit_years:.1f} yr
  Distance (20 km/visit):      {total_visits*20:,} km ({km_equator:.1f} equator laps)

AGE-STRATIFIED (supplementary, Figure S2; n with valid DOB = {n_age:,})"""
for stratum, r in age_results.items():
    med_g = ('Not reached' if np.isinf(r['median_generator'])
             else f"{r['median_generator']:.2f} yr")
    summary += f"""
  {stratum}
    n={r['n']}, deaths={r['deaths']}, replacements={r['replacements']}
    Median patient survival:      {r['median_patient']:.2f} yr
    Median generator (CIF=0.5):   {med_g}
    Deaths without replacement:   {r['deaths_no_rep']}/{r['deaths']} ({r['pct_deaths_no_rep']:.0f}%)
    At 9 yr: patient S={r['S_pac_9']*100:.0f}%, gen freedom={r['freedom_gen_9']*100:.0f}%, Δ={(r['freedom_gen_9']-r['S_pac_9'])*100:.0f} pp"""

summary += f"""

  Log-rank (patient survival, <75 vs ≥75): p = {lr_age.p_value:.2e}
  Excluded from age analysis (missing DOB): {n_age_missing}

SEX-STRATIFIED (supplementary, Figure S3)"""
for sex_key, r in sex_results.items():
    med_g = ('Not reached' if np.isinf(r['median_generator'])
             else f"{r['median_generator']:.2f} yr")
    summary += f"""
  {sex_key} ({r['label_es']})
    n={r['n']}, deaths={r['deaths']}, replacements={r['replacements']}
    Median patient survival:      {r['median_patient']:.2f} yr (95% CI {r['median_patient_ci'][0]:.2f}-{r['median_patient_ci'][1]:.2f})
    Median generator (CIF=0.5):   {med_g}
    Deaths without replacement:   {r['deaths_no_rep']}/{r['deaths']} ({r['pct_deaths_no_rep']:.0f}%)
    At 9 yr: patient S={r['S_pac_9']*100:.0f}%, gen freedom={r['freedom_gen_9']*100:.0f}%, Δ={(r['freedom_gen_9']-r['S_pac_9'])*100:.0f} pp"""

summary += f"""

  Log-rank (patient survival, M vs F):      p = {lr_sex_pac.p_value:.3f}
  Log-rank (replacement hazard, M vs F):    p = {lr_sex_gen.p_value:.3f}

============================================================
"""

print(summary)
with open('results_summary_v20.txt', 'w') as f:
    f.write(summary)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 1 — PATIENT KM vs GENERATOR CIF
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 8.5), facecolor='white')
gs  = fig.add_gridspec(2, 1, height_ratios=[3.2, 0.75], hspace=0.06)
ax  = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])


# Patient curve with CI
def _fill_ci_km(ax_, kmf, color, alpha=0.12):
    ci = kmf.confidence_interval_
    ax_.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                     color=color, alpha=alpha, linewidth=0, step='post')


_fill_ci_km(ax, kmf_pac, COL_PAC)
ax.step(kmf_pac.timeline,
        kmf_pac.survival_function_.values.flatten(),
        where='post', color=COL_PAC, linewidth=2.3, zorder=5)

# Generator Aalen-Johansen: freedom = 1 - CIF
ci_gen = aj_gen.confidence_interval_
t_gen_grid = ci_gen.index.values
freedom_lo = 1 - ci_gen.iloc[:, 1].values   # note: flip bounds
freedom_hi = 1 - ci_gen.iloc[:, 0].values
freedom = 1 - aj_gen.cumulative_density_.iloc[:, 0].values

ax.fill_between(t_gen_grid, freedom_lo, freedom_hi,
                color=COL_GEN, alpha=0.12, linewidth=0, step='post')
ax.step(aj_gen.cumulative_density_.index,
        freedom,
        where='post', color=COL_GEN, linewidth=2.3, zorder=5)

# Median patient marker
ax.plot([med_pac, med_pac], [0, 0.5], color=COL_PAC, lw=1.1, ls='--', alpha=0.7)
ax.plot([0, med_pac], [0.5, 0.5], color='grey', lw=0.8, ls=':', alpha=0.6)
ax.annotate(
    f'Median patient survival\n{med_pac:.1f} yr (95% CI {ci_lo_pac:.1f}–{ci_hi_pac:.1f})',
    xy=(med_pac, 0.5), xytext=(med_pac + 0.9, 0.61),
    fontsize=8.5, color=COL_PAC,
    arrowprops=dict(arrowstyle='->', color=COL_PAC, lw=1.0),
)

# Gap at 9 yr
sp9 = float(kmf_pac.survival_function_at_times(9).values[0])
sg9 = aj_freedom(aj_gen, 9)
ax.annotate('', xy=(9, sg9), xytext=(9, sp9),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.4))
ax.text(9.3, (sg9 + sp9) / 2,
        f'Δ = {(sg9 - sp9) * 100:.0f} pp\nat 9 yr',
        fontsize=8.5, color='#444', va='center')

# Stats box
props = dict(boxstyle='round,pad=0.55',
             facecolor='#f4f4f4', edgecolor='#c8c8c8', alpha=0.96)
box_txt = (
    f'N = {n:,}  ·  Follow-up 1997–2025\n'
    f'Deaths: {df["obito"].sum():,} ({df["obito"].mean()*100:.0f}%)     '
    f'Replacements: {df["recambio"].sum():,} ({df["recambio"].mean()*100:.0f}%)\n'
    f'Died without elective replacement: '
    f'{died_orig:,} ({died_orig/df["obito"].sum()*100:.0f}% of deaths)'
)
ax.text(0.015, 0.055, box_txt, transform=ax.transAxes, fontsize=8.3,
        va='bottom', bbox=props, family='monospace')

legend_elements = [
    Line2D([0], [0], color=COL_PAC, lw=2.3, label='Patient survival (Kaplan–Meier)'),
    Line2D([0], [0], color=COL_GEN, lw=2.3,
           label='Freedom from replacement (1 − Aalen-Johansen CIF)'),
    mpatches.Patch(facecolor='grey', alpha=0.22, label='95% CI'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
          framealpha=0.93, edgecolor='#ccc')

ax.set_title(
    'Who outlives whom: patient versus pacemaker generator\n'
    'Patient survival (Kaplan–Meier) and generator freedom from replacement\n'
    '(Aalen-Johansen cumulative incidence function, death as competing event)',
    fontsize=11.5, fontweight='bold', pad=12)
ax.set_ylabel('Survival probability / freedom from replacement', fontsize=10)
ax.set_xlim(0, 22); ax.set_ylim(0, 1.04)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.grid(axis='y', color='#e0e0e0', lw=0.7)
ax.grid(axis='x', color='#eeeeee', lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticklabels([])

# Number at risk table
ax2.set_xlim(ax.get_xlim()); ax2.axis('off')
risk_times = list(range(0, 23, 2))
nrisk_pac = [(df['t_pac'] >= t).sum() for t in risk_times]
# For the competing-risks endpoint, at-risk = not yet had any event
nrisk_gen = [((df['t_cr'] >= t)).sum() for t in risk_times]

ax2.text(-1.2, 0.75, 'Patient',    color=COL_PAC, fontweight='bold', fontsize=9.5)
ax2.text(-1.2, 0.42, 'Generator',  color=COL_GEN, fontweight='bold', fontsize=9.5)
ax2.text(-1.2, 0.92, 'Number at risk', style='italic', color='#555', fontsize=9)
for t, np_, ng_ in zip(risk_times, nrisk_pac, nrisk_gen):
    ax2.text(t, 0.75, f'{np_:,}', color=COL_PAC, fontsize=8.5, ha='center')
    ax2.text(t, 0.42, f'{ng_:,}', color=COL_GEN, fontsize=8.5, ha='center')
    ax2.text(t, 0.10, f'{t}',     color='#666',  fontsize=9,   ha='center')

ax2.text(11, -0.3, 'Time from implantation (years)',
         fontsize=10, ha='center', color='#333')

plt.savefig('figure1_km_patient_vs_generator.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 1 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. FIGURE 2 — PARADOX OF THE FREQUENT VISITOR
# ─────────────────────────────────────────────────────────────────────────────

fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.5),
                                 gridspec_kw={'width_ratios': [1, 1]},
                                 facecolor='white')

# Panel A: KM by quartile
for q, color in zip(QTILES, QCOLORS):
    kmf = kmfs_q[q]
    ci = kmf.confidence_interval_
    axA.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                     color=color, alpha=0.12, linewidth=0, step='post')
    axA.step(kmf.timeline,
             kmf.survival_function_.values.flatten(),
             where='post', color=color, linewidth=2.2, label=q)
    # Median marker
    if not np.isinf(kmf.median_survival_time_):
        axA.plot(kmf.median_survival_time_, 0.5, 'o', color=color,
                 markersize=7, zorder=6)

axA.set_title('A — Kaplan–Meier survival by\nnumber of follow-up visits',
              fontsize=12, fontweight='bold', pad=10)
axA.set_xlabel('Time from implantation (years)', fontsize=10)
axA.set_ylabel('Survival probability', fontsize=10)
axA.set_xlim(0, 22); axA.set_ylim(0, 1.04)
axA.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
axA.xaxis.set_major_locator(MultipleLocator(4))
axA.legend(loc='upper right', fontsize=9.5, edgecolor='#ccc')
axA.grid(axis='y', color='#e0e0e0', lw=0.7)
axA.grid(axis='x', color='#eeeeee', lw=0.5)
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)

# Subtitle box
q1_mort = visit_df.iloc[0]['crude_mort_pct']
q4_mort = visit_df.iloc[-1]['crude_mort_pct']
axA.text(0.02, 0.02,
         f'Crude mortality: {q1_mort:.0f}% (Q1) vs {q4_mort:.0f}% (Q4)\n'
         f'→ see Panel B for time-adjusted rates',
         transform=axA.transAxes, fontsize=8.8,
         color='#555', style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffacd',
                   edgecolor='#ccc', alpha=0.85))

# Panel B: rates
y_pos = np.arange(len(visit_df))
colors_rev = QCOLORS
rates = visit_df['rate_per_100py'].values
for i, (rate, q, color) in enumerate(zip(rates, QTILES, colors_rev)):
    axB.barh(i, rate, color=color, edgecolor='white', height=0.65)
    axB.text(rate + 0.7, i, f'{rate:.1f}',
             va='center', fontsize=11, fontweight='bold', color=color)
    sub_row = visit_df.iloc[i]
    axB.text(0.8, i - 0.32,
             f'n={sub_row["n"]:.0f}  ·  {sub_row["deaths"]:.0f} deaths  ·  '
             f'{sub_row["pt_years"]:.0f} pt-yr',
             va='center', fontsize=8.3, color='#555')

axB.set_yticks(y_pos); axB.set_yticklabels(QTILES, fontsize=10)
axB.set_xlabel('Deaths per 100 patient-years', fontsize=10)
axB.set_title('B — Mortality rate adjusted\nfor time under observation',
              fontsize=12, fontweight='bold', pad=10)
axB.set_xlim(0, max(rates) * 1.25)
axB.grid(axis='x', color='#e0e0e0', lw=0.7)
axB.spines['top'].set_visible(False)
axB.spines['right'].set_visible(False)

ratio = rates[0] / rates[-1]
axB.text(0.98, 0.5,
         f'Gradient persists\nafter adjustment\n(~{ratio:.0f}× Q1 → Q4)',
         transform=axB.transAxes, fontsize=9, color='#555',
         style='italic', ha='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                   edgecolor='#ccc', alpha=0.9))

fig2.suptitle(
    f'Figure 2.  The paradox of the frequent visitor\n'
    f'Follow-up visit frequency and long-term survival in {n:,} pacemaker recipients',
    fontsize=12.5, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('figure2_paradox_frequent_visitor.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 2 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FIGURE 3 — RACE OUTCOMES
# ─────────────────────────────────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(11, 6), facecolor='white')
labels = ['Died with original\ngenerator in place',
          'Alive with original\ngenerator in place',
          'Died after\nreplacement',
          'Alive after\nreplacement']
values = [died_orig, vivo_orig, died_recam, vivo_recam]
pct    = [v/n*100 for v in values]
colors = ['#C0392B', '#E8A9A0', '#34495E', '#9BC5E0']

bars = ax3.bar(labels, pct, color=colors, edgecolor='white', linewidth=1.5)
for bar, p, v in zip(bars, pct, values):
    ax3.text(bar.get_x() + bar.get_width()/2,
             p + 1.5, f'{p:.1f}%\n(n={v})',
             ha='center', fontsize=11, fontweight='bold')

ax3.set_ylabel('Percentage of cohort (%)', fontsize=11)
ax3.set_ylim(0, max(pct) * 1.25)
ax3.set_title(
    f'Figure 3.  Race outcomes: classification of all {n:,} pacemaker recipients\n'
    f'at registry closure (1 October 2025)',
    fontsize=12.5, fontweight='bold', pad=12)
ax3.grid(axis='y', color='#e0e0e0', lw=0.7)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figure3_race_outcomes.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 3 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 11. FIGURE S1 — STROBE FLOW DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

figS1, axS1 = plt.subplots(figsize=(10, 9), facecolor='white')
axS1.set_xlim(0, 10); axS1.set_ylim(0, 11); axS1.axis('off')


def _flow_box(cx, cy, w, h, text, fc='#e8f0f7', ec='#2c3e50',
              fontweight='normal', fontsize=10.5):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle='round,pad=0.06',
                         facecolor=fc, edgecolor=ec, linewidth=1.2)
    axS1.add_patch(box)
    axS1.text(cx, cy, text, ha='center', va='center',
              fontsize=fontsize, fontweight=fontweight, wrap=True)


from matplotlib.patches import FancyBboxPatch


def _flow_arrow(x1, y1, x2, y2, color='#2c3e50', lw=1.3):
    axS1.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>',
        mutation_scale=15, color=color, linewidth=lw))


axS1.text(5, 10.4, 'Figure S1.  STROBE Flow Diagram',
          ha='center', va='center', fontsize=13, fontweight='bold')
axS1.text(5, 9.95,
          'Patient flow from institutional pacing registry to analytic cohort',
          ha='center', va='center', fontsize=10, style='italic', color='#555')

main_x = 3.3
_flow_box(main_x, 8.4, 5.6, 1.1,
          f'Patients in institutional pacing registry\n'
          f'(September 1997 – September 2025)\n'
          f'n = {n_registry_pm:,}',
          fc='#e8f0f7', fontweight='bold', fontsize=10.5)
_flow_arrow(main_x, 7.85, main_x, 7.05)

_flow_box(main_x, 6.4, 5.6, 1.2,
          f'Analytic cohort: first permanent pacemaker with\n'
          f'valid follow-up time for both endpoints\n'
          f'n = {n:,}',
          fc='#dde8f3', fontweight='bold', fontsize=10.5)

_flow_box(7.6, 7.4, 3.9, 1.4,
          f'Excluded (n = {n_excluded})\n\n'
          f'• Deceased without recorded last\n'
          f'  clinical contact date: n = {n_excl_no_uc}\n'
          f'• Other invalid follow-up time: n = {n_excl_other}',
          fc='#fbeaea', ec='#a04040', fontsize=9.5)
_flow_arrow(main_x + 2.8, 7.4, 7.6 - 1.95, 7.4, color='#a04040', lw=1.1)

_flow_arrow(main_x, 5.8, main_x, 5.0)
_flow_box(main_x, 4.3, 5.6, 1.4,
          f'Outcomes ascertained per patient\n\n'
          f'Patient endpoint: {df["obito"].sum()} deaths  ·  '
          f'{(~df["obito"]).sum()} censored alive\n'
          f'Generator endpoint: {df["recambio"].sum()} replaced  ·  '
          f'{df["obito"].sum() - died_recam} competing death  ·  '
          f'{vivo_orig} censored alive',
          fc='#f5f5f5', fontsize=9.6)

_flow_arrow(main_x, 3.6, main_x, 2.8)
_flow_box(main_x, 2.1, 5.6, 1.4,
          f'Survival analysis\n\n'
          f'Patient: Kaplan–Meier\n'
          f'Generator: Aalen-Johansen CIF, death as competing event',
          fc='#dde8f3', fontweight='bold', fontsize=10)

axS1.text(5, 0.85,
          f'{df["t_pac"].sum():.0f} patient-years total  ·  '
          f'median follow-up {df["t_pac"].median():.1f} years '
          f'(IQR {df["t_pac"].quantile(.25):.1f}–{df["t_pac"].quantile(.75):.1f})  ·  '
          f'median age at implantation '
          f'{df["edad"].median():.1f} years '
          f'(IQR {df["edad"].quantile(.25):.1f}–{df["edad"].quantile(.75):.1f})',
          ha='center', va='center', fontsize=8.8, color='#444')
axS1.text(5, 0.35,
          'Patients with prior device implantation elsewhere were excluded '
          'by registry inclusion criteria (this exclusion preceded data extraction).',
          ha='center', va='center', fontsize=8.2, color='#666', style='italic')

plt.tight_layout()
plt.savefig('figureS1_strobe_flow.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure S1 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 12. FIGURE S2 — AGE-STRATIFIED (<75 vs ≥75)
# ─────────────────────────────────────────────────────────────────────────────

figS2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6),
                                  facecolor='white', sharey=True)
figS2.subplots_adjust(wspace=0.08)

for label, axx in [('<75 yr', axL), ('≥75 yr', axR)]:
    r    = age_results[label]
    kmfP = r['kmf_pac']
    ajG  = r['aj_gen']

    # Patient KM
    ci = kmfP.confidence_interval_
    axx.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                     color=COL_PAC, alpha=0.12, linewidth=0, step='post')
    axx.step(kmfP.timeline,
             kmfP.survival_function_.values.flatten(),
             where='post', color=COL_PAC, linewidth=2.3,
             zorder=5, label='Patient survival')

    # Generator freedom (1 - CIF)
    cig = ajG.confidence_interval_
    tg = cig.index.values
    freedom_lo_s = 1 - cig.iloc[:, 1].values
    freedom_hi_s = 1 - cig.iloc[:, 0].values
    freedom_s = 1 - ajG.cumulative_density_.iloc[:, 0].values
    axx.fill_between(tg, freedom_lo_s, freedom_hi_s,
                     color=COL_GEN, alpha=0.12, linewidth=0, step='post')
    axx.step(ajG.cumulative_density_.index, freedom_s,
             where='post', color=COL_GEN, linewidth=2.3,
             zorder=5, label='Freedom from replacement')

    # Gap at 9 yr
    axx.annotate('', xy=(9, r['freedom_gen_9']), xytext=(9, r['S_pac_9']),
                 arrowprops=dict(arrowstyle='<->', color='#555', lw=1.3))
    axx.text(9.5, (r['freedom_gen_9'] + r['S_pac_9']) / 2,
             f'Δ = {(r["freedom_gen_9"] - r["S_pac_9"]) * 100:.0f} pp\nat 9 yr',
             fontsize=8.8, color='#444', va='center', zorder=7)

    subtitle = (
        f"n = {r['n']:,}  ·  deaths: {r['deaths']}  ·  replacements: {r['replacements']}\n"
        f"{r['deaths_no_rep']}/{r['deaths']} deaths "
        f"({r['pct_deaths_no_rep']:.0f}%) without elective replacement"
    )
    axx.set_title(f'Age at implant: {label}\n{subtitle}',
                  fontsize=10.5, fontweight='bold', pad=10)
    axx.set_xlabel('Time from implantation (years)', fontsize=10)
    axx.set_xlim(0, 22); axx.set_ylim(0, 1.04)
    axx.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axx.xaxis.set_major_locator(MultipleLocator(4))
    axx.grid(axis='y', color='#e0e0e0', lw=0.7)
    axx.grid(axis='x', color='#eeeeee', lw=0.5)
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)
    if label == '<75 yr':
        axx.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#ccc')
        axx.set_ylabel('Survival probability / freedom from replacement',
                       fontsize=10)

figS2.suptitle(
    'Figure S2.  Patient versus generator survival, stratified by age at implantation\n'
    f'Analysis restricted to {n_age:,} patients with valid date of birth',
    fontsize=12, fontweight='bold', y=1.03)

plt.savefig('figureS2_age_stratified.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure S2 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 13. FIGURE S3 — SEX-STRATIFIED
# ─────────────────────────────────────────────────────────────────────────────

figS3, (axL3, axR3) = plt.subplots(1, 2, figsize=(13, 6),
                                    facecolor='white', sharey=True)
figS3.subplots_adjust(wspace=0.08)

for label_en, axx in [('Men', axL3), ('Women', axR3)]:
    r    = sex_results[label_en]
    kmfP = r['kmf_pac']
    ajG  = r['aj_gen']

    ci = kmfP.confidence_interval_
    axx.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                     color=COL_PAC, alpha=0.12, linewidth=0, step='post')
    axx.step(kmfP.timeline,
             kmfP.survival_function_.values.flatten(),
             where='post', color=COL_PAC, linewidth=2.3,
             zorder=5, label='Patient survival')

    cig = ajG.confidence_interval_
    tg = cig.index.values
    freedom_lo_s = 1 - cig.iloc[:, 1].values
    freedom_hi_s = 1 - cig.iloc[:, 0].values
    freedom_s = 1 - ajG.cumulative_density_.iloc[:, 0].values
    axx.fill_between(tg, freedom_lo_s, freedom_hi_s,
                     color=COL_GEN, alpha=0.12, linewidth=0, step='post')
    axx.step(ajG.cumulative_density_.index, freedom_s,
             where='post', color=COL_GEN, linewidth=2.3,
             zorder=5, label='Freedom from replacement')

    axx.annotate('', xy=(9, r['freedom_gen_9']), xytext=(9, r['S_pac_9']),
                 arrowprops=dict(arrowstyle='<->', color='#555', lw=1.3))
    axx.text(9.5, (r['freedom_gen_9'] + r['S_pac_9']) / 2,
             f'Δ = {(r["freedom_gen_9"] - r["S_pac_9"]) * 100:.0f} pp\nat 9 yr',
             fontsize=8.8, color='#444', va='center', zorder=7)

    subtitle = (
        f"n = {r['n']:,}  ·  deaths: {r['deaths']}  ·  replacements: {r['replacements']}\n"
        f"{r['deaths_no_rep']}/{r['deaths']} deaths "
        f"({r['pct_deaths_no_rep']:.0f}%) without elective replacement"
    )
    axx.set_title(f'Sex: {label_en}\n{subtitle}',
                  fontsize=10.5, fontweight='bold', pad=10)
    axx.set_xlabel('Time from implantation (years)', fontsize=10)
    axx.set_xlim(0, 22); axx.set_ylim(0, 1.04)
    axx.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axx.xaxis.set_major_locator(MultipleLocator(4))
    axx.grid(axis='y', color='#e0e0e0', lw=0.7)
    axx.grid(axis='x', color='#eeeeee', lw=0.5)
    axx.spines['top'].set_visible(False)
    axx.spines['right'].set_visible(False)
    if label_en == 'Men':
        axx.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#ccc')
        axx.set_ylabel('Survival probability / freedom from replacement',
                       fontsize=10)

figS3.suptitle(
    'Figure S3.  Patient versus generator survival, stratified by sex\n'
    f'Aalen-Johansen estimates, {n:,} pacemaker recipients',
    fontsize=12, fontweight='bold', y=1.03)

plt.savefig('figureS3_sex_stratified.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure S3 saved.")


print("\nAll outputs generated successfully.")
print("Files:")
print("  results_summary_v20.txt")
print("  figure1_km_patient_vs_generator.png")
print("  figure2_paradox_frequent_visitor.png")
print("  figure3_race_outcomes.png")
print("  figureS1_strobe_flow.png")
print("  figureS2_age_stratified.png")
print("  figureS3_sex_stratified.png")
