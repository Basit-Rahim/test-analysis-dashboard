import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

# ─── CTT ─────────────────────────────────────────────────────────────────────

def compute_ctt(df):
    df_num = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    cols = df_num.columns.tolist()
    if not cols:
        return None, None, None, None
    total = df_num.sum(axis=1)
    k = len(cols)
    stats = []
    for col in cols:
        item = df_num[col]
        rest = total - item
        try:
            corr = item.corr(rest)
        except Exception:
            corr = np.nan
        stats.append({
            'Item': col,
            'Difficulty (p)': round(item.mean(), 3),
            'Std Dev': round(item.std(ddof=1), 3),
            'Item-Total r': round(corr, 3) if not np.isnan(corr) else np.nan,
            'Missing': int(item.isna().sum()),
        })
    stats_df = pd.DataFrame(stats).set_index('Item')
    item_vars = df_num.var(axis=0, ddof=1)
    total_var = total.var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var) if total_var > 0 and k > 1 else np.nan
    return df_num, total, stats_df, alpha


# ─── IRT ─────────────────────────────────────────────────────────────────────

def _initial_theta(data, k):
    scores = data.sum(axis=1).astype(float)
    scores = np.clip(scores, 0.5, k - 0.5)
    theta = np.log(scores / (k - scores))
    return (theta - theta.mean()) / max(theta.std(), 1e-9)


def estimate_irt(df_num, model):
    data = df_num.values.astype(float)
    n, k = data.shape
    theta = _initial_theta(data, k)
    params = {}

    for i, col in enumerate(df_num.columns):
        y = data[:, i]
        p_bar = np.clip(y.mean(), 0.01, 0.99)
        b0 = float(-np.log(p_bar / (1 - p_bar)))

        if model == '1PL':
            def nll(x):
                prob = np.clip(1 / (1 + np.exp(-(theta - x[0]))), 1e-9, 1 - 1e-9)
                return -np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
            res = minimize(nll, [b0], method='L-BFGS-B', bounds=[(-4, 4)])
            params[col] = dict(a=1.0, b=float(res.x[0]), c=0.0)

        elif model == '2PL':
            def nll(x):
                prob = np.clip(1 / (1 + np.exp(-x[0] * (theta - x[1]))), 1e-9, 1 - 1e-9)
                return -np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
            res = minimize(nll, [1.0, b0], method='L-BFGS-B', bounds=[(0.1, 4), (-4, 4)])
            params[col] = dict(a=float(res.x[0]), b=float(res.x[1]), c=0.0)

        else:  # 3PL
            def nll(x):
                prob = np.clip(x[2] + (1 - x[2]) / (1 + np.exp(-x[0] * (theta - x[1]))), 1e-9, 1 - 1e-9)
                return -np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
            res = minimize(nll, [1.0, b0, 0.2], method='L-BFGS-B',
                           bounds=[(0.1, 4), (-4, 4), (0.0, 0.35)])
            params[col] = dict(a=float(res.x[0]), b=float(res.x[1]), c=float(res.x[2]))

    return params


COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
          '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']


def make_item_fig(item, p, model, color):
    theta_range = np.linspace(-4, 4, 300)
    a, b, c = p['a'], p['b'], p['c']
    prob = c + (1 - c) / (1 + np.exp(-a * (theta_range - b)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theta_range, y=prob,
        mode='lines',
        line=dict(color=color, width=2.5),
        showlegend=False,
    ))
    fig.add_vline(x=b, line_dash='dot', line_color='gray', line_width=1)
    fig.add_hline(y=0.5, line_dash='dash', line_color='lightgray', line_width=1)

    label = f"a={a:.2f}  b={b:.2f}"
    if model == '3PL':
        label += f"  c={c:.2f}"

    fig.update_layout(
        height=240,
        margin=dict(l=40, r=16, t=36, b=36),
        title=dict(text=label, font=dict(size=11), x=0.5, xanchor='center'),
        xaxis=dict(title='θ', range=[-4, 4], gridcolor='#eee', zeroline=False),
        yaxis=dict(title='P(correct)', range=[0, 1], gridcolor='#eee', zeroline=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig


def render_icc_cards(params, model):
    items = list(params.keys())
    cols_per_row = 3
    for row_start in range(0, len(items), cols_per_row):
        row_items = items[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col_idx, item in enumerate(row_items):
            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"**{item}**")
                    color = COLORS[(row_start + col_idx) % len(COLORS)]
                    fig = make_item_fig(item, params[item], model, color)
                    st.plotly_chart(fig, width='stretch')


# ─── APP ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Test Analysis Dashboard', layout='wide')
st.title('Test Analysis Dashboard')

# ── Top-level framework selector
framework = st.selectbox(
    'Analysis Framework',
    ['Classical Test Theory (CTT)', 'Item Response Theory (IRT)'],
    index=0,
)

st.divider()

# ── Sidebar: data loading
with st.sidebar:
    st.header('Data')
    uploaded = st.file_uploader('Upload CSV (rows = respondents, cols = items)', type=['csv'])
    load_sample = st.button('Load sample_responses.csv')
    st.markdown('---')
    st.caption('CSV format: each row is one respondent, each column is one item (binary 0/1 or polytomous).')

# Persist df across reruns via session_state
if uploaded is not None:
    try:
        st.session_state['df'] = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f'Error reading CSV: {e}')
elif load_sample:
    try:
        st.session_state['df'] = pd.read_csv('sample_responses.csv')
    except Exception:
        st.error('Could not find sample_responses.csv.')

df = st.session_state.get('df', None)

if df is None:
    st.info('Load data using the sidebar to begin analysis.')
    st.stop()

# ── Data preview
with st.expander('Preview data', expanded=False):
    st.dataframe(df, width='stretch')

# ── Parse numeric
df_num_raw = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
if df_num_raw.empty:
    st.error('No numeric columns found.')
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CTT VIEW
# ═══════════════════════════════════════════════════════════════════════════════
if framework == 'Classical Test Theory (CTT)':
    df_num, total, item_stats, alpha = compute_ctt(df)

    if df_num is None:
        st.error('No numeric item columns found.')
        st.stop()

    n_persons, n_items = df_num.shape

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Respondents', n_persons)
    c2.metric('Items', n_items)
    c3.metric("Cronbach's α", f'{alpha:.3f}' if not np.isnan(alpha) else 'N/A')
    c4.metric('Mean Score', f'{total.mean():.2f}')

    st.subheader('Item Statistics Matrix')
    st.dataframe(
        item_stats.style.background_gradient(subset=['Difficulty (p)'], cmap='Blues')
                        .background_gradient(subset=['Item-Total r'], cmap='RdYlGn')
                        .format(precision=3),
        width='stretch',
    )

    csv_item = item_stats.reset_index().to_csv(index=False).encode()
    st.download_button('Download item stats CSV', csv_item, 'item_stats.csv', 'text/csv')

    st.subheader('Respondent Total Scores')
    score_df = pd.concat([df.reset_index(drop=True),
                          total.rename('Total Score').reset_index(drop=True)], axis=1)
    st.dataframe(score_df, width='stretch')

    st.subheader('Score Distribution')
    hist_vals = total.value_counts().sort_index()
    fig_hist = go.Figure(go.Bar(
        x=hist_vals.index, y=hist_vals.values,
        marker_color='#636EFA',
        text=hist_vals.values, textposition='outside',
    ))
    fig_hist.update_layout(
        xaxis_title='Total Score', yaxis_title='Frequency',
        plot_bgcolor='white', height=320,
    )
    fig_hist.update_xaxes(gridcolor='#eee')
    fig_hist.update_yaxes(gridcolor='#eee')
    st.plotly_chart(fig_hist, width='stretch')

    csv_scores = score_df.to_csv(index=False).encode()
    st.download_button('Download respondent scores CSV', csv_scores, 'respondent_scores.csv', 'text/csv')

# ═══════════════════════════════════════════════════════════════════════════════
# IRT VIEW
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader('IRT Configuration')
    irt_model = st.radio(
        'IRT Model (number of parameters per item)',
        ['1PL', '2PL', '3PL'],
        index=1,
        horizontal=True,
        captions=[
            '1PL — Rasch: difficulty only (b)',
            '2PL — difficulty + discrimination (a, b)',
            '3PL — difficulty + discrimination + guessing (a, b, c)',
        ],
    )

    st.markdown("""
| Model | Parameters | Description |
|-------|-----------|-------------|
| **1PL** | b | Difficulty only; all items discriminate equally |
| **2PL** | a, b | Discrimination + difficulty; most common in practice |
| **3PL** | a, b, c | Adds pseudo-guessing; suited for MCQ tests |
""")

    run = st.button('Estimate Parameters & Plot ICC', type='primary')

    if run:
        with st.spinner(f'Estimating {irt_model} parameters…'):
            try:
                params = estimate_irt(df_num_raw, irt_model)
                st.session_state['irt_params'] = params
                st.session_state['irt_model'] = irt_model
            except Exception as e:
                st.error(f'Estimation failed: {e}')
                st.stop()

    # Show results if params exist in session (persists after button click rerun)
    params = st.session_state.get('irt_params')
    stored_model = st.session_state.get('irt_model')

    if params is not None:
        st.subheader('Estimated Item Parameters')
        rows_p = []
        for item, p in params.items():
            row = {'Item': item, 'b (Difficulty)': round(p['b'], 3),
                   'a (Discrimination)': round(p['a'], 3) if stored_model != '1PL' else '—'}
            if stored_model == '3PL':
                row['c (Guessing)'] = round(p['c'], 3)
            rows_p.append(row)
        param_df = pd.DataFrame(rows_p).set_index('Item')

        st.dataframe(
            param_df.style.background_gradient(subset=['b (Difficulty)'], cmap='RdYlGn_r'),
            width='stretch',
        )
        csv_params = param_df.reset_index().to_csv(index=False).encode()
        st.download_button('Download parameters CSV', csv_params, 'irt_params.csv', 'text/csv')

        st.subheader('Item Characteristic Curves (ICC)')
        st.caption(
            'Each curve shows the probability of a correct response as a function of ability (θ). '
            'The dotted vertical line marks the difficulty parameter **b** (θ where P = 0.5 for 1PL/2PL).'
        )
        render_icc_cards(params, stored_model)

        st.subheader('Test Characteristic Curve (TCC)')
        theta_range = np.linspace(-4, 4, 300)
        tcc = np.zeros(300)
        for p in params.values():
            tcc += p['c'] + (1 - p['c']) / (1 + np.exp(-p['a'] * (theta_range - p['b'])))
        fig_tcc = go.Figure(go.Scatter(
            x=theta_range, y=tcc,
            mode='lines', line=dict(color='#636EFA', width=2.5),
        ))
        fig_tcc.update_layout(
            xaxis_title='θ (Ability)', yaxis_title='Expected Score',
            plot_bgcolor='white', height=340,
        )
        fig_tcc.update_xaxes(gridcolor='#eee')
        fig_tcc.update_yaxes(gridcolor='#eee')
        st.plotly_chart(fig_tcc, width='stretch')
    else:
        st.info('Press **Estimate Parameters & Plot ICC** to run the analysis.')
