import optuna
import itertools
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from optuna.samplers import NSGAIISampler
from count_data_infectious import get_peak_infectious
from count_data_susceptible import get_lockdown_days
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

COLORS_HEX = [
    '#4C9BE8', '#E8714C', '#4CE8A0', '#A04CE8',
    '#E8C44C', '#E84CA0', '#4CE8E8', '#E84C4C', '#8AE84C',
]

def run_optimization(
    setup_simulator_fn,
    epidemic_days,
    vacc_days_list,
    vacc_eff_list,
    vacc_intensity_list,
    capacity=3000,
    beta=1.0,
    n_runs=3,
    n_trials=150,
    population_size=30,
    num_populs=3,
    num_sites=2,
    vacc_susceptibility_type=2,
    base_transmission_rate = 0.25,
    output_html="pareto_fronts.html",
    output_importance_html="param_importance.html",
    output_importance_png="param_importance.png",
    robustness_runs=0,
    output_robustness_html="robustness.html",
    output_robustness_png="robustness.png",
):
    """
    Параметры
    ---------
    setup_simulator_fn      : callable (x_close, y_open, lock_dens, seed) -> sim
                              БЕЗ параметров вакцины — они добавляются внутри
    epidemic_days           : int
    vacc_days_list          : list[int]   — день изобретения вакцины
    vacc_eff_list           : list[float] — эффективность вакцины (0..1)
    vacc_intensity_list     : list[float] — скорость вакцинации (rate 0->vacc_susceptibility_type)
    capacity                : int   — пропускная способность здравоохранения
    beta                    : float — вес жёсткости локдауна в M2
    n_runs                  : int   — прогонов для усреднения стохастичности
    n_trials                : int   — триалов optuna на сценарий
    population_size         : int   — размер популяции NSGA-II
    num_populs / num_sites  : int
    vacc_susceptibility_type: int   — susceptibility_type вакцинированных (default 2)
    output_html             : str   — Pareto HTML
    output_importance_html  : str   — важность параметров HTML
    output_importance_png   : str   — важность параметров PNG
    robustness_runs         : int   — доп. прогонов для анализа робастности (0 = выкл)
    output_robustness_html  : str   — робастность HTML
    output_robustness_png   : str   — робастность PNG

    Возвращает
    ----------
    all_results : dict { (vacc_day, eff, intensity): list[FrozenTrial] }
    """
    scenarios   = list(itertools.product(vacc_days_list, vacc_eff_list, vacc_intensity_list))
    all_results = {}
    all_studies = {}

    for vacc_day, vacc_eff, vacc_intensity in tqdm(scenarios, desc="сценарии вакцины"):
        label = f"день={vacc_day}, эфф={vacc_eff:.0%}, инт={vacc_intensity:.3f}"
        print(f"\n[оптимизация] {label}")

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=NSGAIISampler(population_size=population_size, seed=42),
        )
        study.optimize(
            _make_objective(
                setup_simulator_fn, epidemic_days,
                vacc_day, vacc_eff, vacc_intensity,
                capacity, beta, n_runs, num_populs, num_sites,
                vacc_susceptibility_type, base_transmission_rate
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        all_results[(vacc_day, vacc_eff, vacc_intensity)] = study.best_trials
        all_studies[(vacc_day, vacc_eff, vacc_intensity)] = study
        print(f"  -> {len(study.best_trials)} Pareto-оптимальных стратегий")

    _plot_pareto(all_results, output_html)
    _plot_importance(all_studies, output_importance_html, output_importance_png)

    print(f"\nГрафики сохранены:")
    print(f"Pareto:    {output_html}")
    print(f"Важность:  {output_importance_html}, {output_importance_png}")

    if robustness_runs > 0:
        print(f"\n[робастность] {robustness_runs} прогонов на точку Pareto-фронта...")
        _analyze_robustness(
            all_results, all_studies,
            setup_simulator_fn, epidemic_days,
            capacity, beta, num_populs, num_sites,
            vacc_susceptibility_type,
            robustness_runs,
            output_robustness_html, output_robustness_png, base_transmission_rate
        )
        print(f"Робастность: {output_robustness_html}, {output_robustness_png}")

    return all_results

def _make_objective(setup_fn, epidemic_days, vacc_day, vacc_eff, vacc_intensity,
                    capacity, beta, n_runs, num_populs, num_sites, vacc_susceptibility_type, base_transmission_rate):
    def objective(trial):
        x_close   = trial.suggest_float("x_close",  0.02, 0.20)
        ratio     = trial.suggest_float("ratio",     0.05, 0.90)
        y_open    = ratio * x_close
        lock_dens = trial.suggest_float("lock_dens", 0.005, 1.0)

        peaks, lockdown_totals = [], []
        for seed in range(n_runs):
            sim = _run_sim_with_vaccine(
                setup_fn, x_close, y_open, lock_dens, seed,
                epidemic_days, vacc_day, vacc_eff, vacc_intensity,
                vacc_susceptibility_type, base_transmission_rate
            )
            raw = get_peak_infectious(sim, num_populs=num_populs,
                                      num_sites=num_sites, days=epidemic_days)
            peaks.append(sum(max(0, v - capacity) for v in raw))
            lockdown_totals.append(
                get_lockdown_days(sim, num_populs=num_populs, days=epidemic_days)
            )

        M1 = sum(peaks) / n_runs
        M2 = (sum(lockdown_totals) / n_runs) * (1 + beta * (1 - lock_dens))
        return M1, M2

    return objective


def _run_sim_with_vaccine(setup_fn, x_close, y_open, lock_dens, seed,
                          total_days, vacc_day, vacc_eff, vacc_intensity,
                          vacc_susceptibility_type, base_transmission_rate):
    sim = setup_fn(x_close, y_open, base_transmission_rate * lock_dens, seed)

    if vacc_day >= total_days:
        sim.simulate(70_000_000, epidemic_time=total_days, method='direct')
        return sim

    # этап 1: до вакцины
    sim.simulate(70_000_000, epidemic_time=vacc_day, method='direct')

    # вводим вакцину
    sim.set_susceptibility(max(0.0, 1.0 - vacc_eff), susceptibility_type=vacc_susceptibility_type)
    for src in range(vacc_susceptibility_type):
        sim.set_immunity_transition(vacc_intensity, source=src, target=vacc_susceptibility_type)

    # этап 2: продолжаем, epidemic_time = оставшиеся дни
    sim.simulate(20_000_000, epidemic_time=total_days, method='direct')
    return sim


def _eval_strategy(setup_fn, params, epidemic_days, vacc_day, vacc_eff, vacc_intensity,
                   capacity, beta, num_populs, num_sites, seeds, vacc_susceptibility_type=2, base_transmission_rate = 0.25):
    m1s, m2s = [], []
    for seed in seeds:
        x_close = params['x_close']
        y_open  = params['ratio'] * x_close
        sim = _run_sim_with_vaccine(
            setup_fn, x_close, y_open, params['lock_dens'],
            seed, epidemic_days, vacc_day, vacc_eff, vacc_intensity,
            vacc_susceptibility_type, base_transmission_rate
        )
        raw  = get_peak_infectious(sim, num_populs=num_populs,
                                   num_sites=num_sites, days=epidemic_days)
        m1   = sum(max(0, v - capacity) for v in raw)
        lock = get_lockdown_days(sim, num_populs=num_populs, days=epidemic_days)
        m2   = lock * (1 + beta * (1 - params['lock_dens']))
        m1s.append(m1)
        m2s.append(m2)
    return m1s, m2s

def _analyze_robustness(all_results, all_studies,
                        setup_fn, epidemic_days,
                        capacity, beta, num_populs, num_sites,
                        vacc_susceptibility_type,
                        robustness_runs,
                        output_html, output_png, base_transmission_rate):
    scenario_data = []

    for idx, ((vacc_day, vacc_eff, vacc_intensity), trials) in enumerate(all_results.items()):
        if not trials:
            continue

        label = f"д={vacc_day} | эфф={vacc_eff:.0%} | инт={vacc_intensity:.3f}"
        print(f"  сценарий: {label}")

        pts_mean_m1, pts_mean_m2 = [], []
        pts_std_m1,  pts_std_m2  = [], []

        sorted_trials = sorted(trials, key=lambda t: t.values[1])

        for t in tqdm(sorted_trials, desc=f"робастность {label}", leave=False):
            seeds = list(range(100, 100 + robustness_runs))
            m1s, m2s = _eval_strategy(
                setup_fn, t.params, epidemic_days,
                vacc_day, vacc_eff, vacc_intensity,
                capacity, beta, num_populs, num_sites, seeds,
                vacc_susceptibility_type,base_transmission_rate
            )
            pts_mean_m1.append(float(np.mean(m1s)))
            pts_mean_m2.append(float(np.mean(m2s)))
            pts_std_m1.append(float(np.std(m1s)))
            pts_std_m2.append(float(np.std(m2s)))

        scenario_data.append({
            'label':    label,
            'color':    COLORS_HEX[idx % len(COLORS_HEX)],
            'vacc_day': vacc_day,
            'vacc_eff': vacc_eff,
            'vacc_int': vacc_intensity,
            'mean_m1':  pts_mean_m1,
            'mean_m2':  pts_mean_m2,
            'std_m1':   pts_std_m1,
            'std_m2':   pts_std_m2,
            'orig_m1':  [t.values[0] for t in sorted_trials],
            'orig_m2':  [t.values[1] for t in sorted_trials],
        })

    _plot_robustness_html(scenario_data, robustness_runs, output_html)
    _plot_robustness_png(scenario_data,  robustness_runs, output_png)


def _plot_robustness_html(scenario_data, robustness_runs, output_html):
    fig  = go.Figure()
    meta = []  # (vacc_day, vacc_eff, vacc_int, trace_idx_start)

    for idx, d in enumerate(scenario_data):
        color = d['color']
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_color = f'rgba({r},{g},{b},0.15)'
        trace_start = idx * 2

        xs_upper = [m2 + s for m2, s in zip(d['mean_m2'], d['std_m2'])]
        xs_lower = [m2 - s for m2, s in zip(d['mean_m2'], d['std_m2'])]

        fig.add_trace(go.Scatter(
            x=xs_upper + xs_lower[::-1],
            y=d['mean_m1'] + d['mean_m1'][::-1],
            fill='toself', fillcolor=fill_color,
            line=dict(width=0), showlegend=False, hoverinfo='skip',
            visible=True,
        ))

        hover = [
            f"<b>{d['label']}</b><br>"
            f"M2 mean: {m2:.1f} ± {sm2:.1f}<br>"
            f"M1 mean: {m1:.0f} ± {sm1:.0f}"
            for m2, sm2, m1, sm1 in zip(
                d['mean_m2'], d['std_m2'], d['mean_m1'], d['std_m1']
            )
        ]
        fig.add_trace(go.Scatter(
            x=d['mean_m2'], y=d['mean_m1'],
            mode='lines+markers',
            name=d['label'],
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color, line=dict(width=1.5, color='white')),
            hovertext=hover, hoverinfo='text',
            visible=True,
        ))

        meta.append((d['vacc_day'], d['vacc_eff'], d['vacc_int'], trace_start))

    n_traces = len(scenario_data) * 2

    def vis_for_key(key, val):
        result = [False] * n_traces
        for m in meta:
            if m[key] == val:
                ts = m[3]
                result[ts] = result[ts+1] = True
        return result

    def vis_all(visible):
        return [visible] * n_traces

    buttons = [
        dict(label="Все",        method="restyle", args=[{"visible": vis_all(True)}]),
        dict(label="Скрыть все", method="restyle", args=[{"visible": vis_all(False)}]),
    ]
    for vd in sorted({m[0] for m in meta}):
        buttons.append(dict(label=f"день {vd}", method="restyle",
                            args=[{"visible": vis_for_key(0, vd)}]))
    for ve in sorted({m[1] for m in meta}):
        buttons.append(dict(label=f"эфф {ve:.0%}", method="restyle",
                            args=[{"visible": vis_for_key(1, ve)}]))
    for vi in sorted({m[2] for m in meta}):
        buttons.append(dict(label=f"инт {vi:.3f}", method="restyle",
                            args=[{"visible": vis_for_key(2, vi)}]))

    fig.update_layout(
        title=dict(
            text=f'<b>Анализ робастности Pareto-фронтов</b><br>'
                 f'<sup>сплошная — среднее по {robustness_runs} прогонам, ',
            font=dict(size=18, color='#2c3e50'),
            x=0.5, xanchor='center',
        ),
        xaxis=dict(
            title='<b>M2</b> — экономические потери',
            gridcolor='rgba(200,200,200,0.3)',
            showline=True, linecolor='#888', mirror=True,
        ),
        yaxis=dict(
            title='<b>M1</b> — перегрузка здравоохранения',
            gridcolor='rgba(200,200,200,0.3)',
            showline=True, linecolor='#888', mirror=True,
        ),
        plot_bgcolor='#f8f9fb', paper_bgcolor='white',
        hovermode='closest',
        hoverlabel=dict(bgcolor='white', bordercolor='#ccc', font_size=12),
        width=1050, height=650,
        legend=dict(x=1.02, y=1, xanchor='left',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#ddd', borderwidth=1, font=dict(size=11)),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.0, y=1.17, xanchor="left",
            pad={"r": 6, "t": 6},
            bgcolor='#eef2f7', bordercolor='#ccd',
            font=dict(size=11), buttons=buttons,
            showactive=True, active=0,
        )],
        margin=dict(t=200, r=220, b=60, l=70),
    )
    fig.write_html(output_html)


def _plot_robustness_png(scenario_data, robustness_runs, output_png):
    fig, ax = plt.subplots(figsize=(10, 6))

    for d in scenario_data:
        color = d['color']
        m2  = np.array(d['mean_m2'])
        m1  = np.array(d['mean_m1'])
        sm2 = np.array(d['std_m2'])
        sm1 = np.array(d['std_m1'])

        ax.fill_betweenx(m1, m2 - sm2, m2 + sm2, alpha=0.15, color=color)
        ax.fill_between(m2, m1 - sm1, m1 + sm1, alpha=0.10, color=color)
        ax.plot(d['orig_m2'], d['orig_m1'],
                linestyle='--', color=color, linewidth=1.2, alpha=0.6)
        ax.plot(m2, m1, 'o-', color=color, linewidth=2.2,
                markersize=7, markeredgecolor='white', markeredgewidth=1.2,
                label=d['label'])

    ax.set_xlabel('M2 — экономические потери', fontsize=12)
    ax.set_ylabel('M1 — перегрузка здравоохранения', fontsize=12)
    ax.set_title(
        f'Анализ робастности Pareto-фронтов\n'
        f'(сплошная = среднее/{robustness_runs} прогонов, пунктир = исходный фронт, полоса = ±std)',
        fontsize=12,
    )
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fb')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()

def _plot_pareto(all_results: dict, output_html: str):
    fig  = go.Figure()
    meta = []

    for idx, ((vacc_day, vacc_eff, vacc_intensity), trials) in enumerate(all_results.items()):
        if not trials:
            continue
        pts   = sorted(trials, key=lambda t: t.values[1])
        xs    = [t.values[1] for t in pts]
        ys    = [t.values[0] for t in pts]
        hover = [
            f"<b>Стратегия</b><br>"
            f"x_close:   {t.params['x_close']:.3f}<br>"
            f"y_open:    {t.params['ratio'] * t.params['x_close']:.3f}"
            f"  (ratio={t.params['ratio']:.3f})<br>"
            f"lock_dens: {t.params['lock_dens']:.3f}<br>"
            f"─────────────<br>"
            f"M1: {t.values[0]:,.0f}<br>"
            f"M2: {t.values[1]:.1f}"
            for t in pts
        ]
        color = COLORS_HEX[idx % len(COLORS_HEX)]
        label = f"д={vacc_day} | эфф={vacc_eff:.0%} | инт={vacc_intensity:.3f}"

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines+markers', name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, line=dict(width=1.5, color='white')),
            hovertext=hover, hoverinfo='text', visible=True,
        ))
        meta.append((vacc_day, vacc_eff, vacc_intensity, idx))

    n = len(meta)

    def vis_for(key, val):
        return [m[key] == val for m in meta]

    buttons = [
        dict(label="Все",        method="restyle", args=[{"visible": [True]  * n}]),
        dict(label="Скрыть все", method="restyle", args=[{"visible": [False] * n}]),
    ]
    for vd in sorted({m[0] for m in meta}):
        buttons.append(dict(label=f"день {vd}",    method="restyle",
                            args=[{"visible": vis_for(0, vd)}]))
    for ve in sorted({m[1] for m in meta}):
        buttons.append(dict(label=f"эфф {ve:.0%}", method="restyle",
                            args=[{"visible": vis_for(1, ve)}]))
    for vi in sorted({m[2] for m in meta}):
        buttons.append(dict(label=f"инт {vi:.3f}", method="restyle",
                            args=[{"visible": vis_for(2, vi)}]))

    fig.update_layout(
        title=dict(text="<b>Pareto-фронты оптимальных стратегий локдауна</b>",
                   font=dict(size=20, color='#2c3e50'), x=0.5, xanchor='center'),
        xaxis=dict(title="<b>M2</b> — экономические потери (дни × жёсткость)",
                   title_font=dict(size=13), gridcolor='rgba(200,200,200,0.3)',
                   showline=True, linecolor='#888', mirror=True),
        yaxis=dict(title="<b>M1</b> — перегрузка здравоохранения (чел·дней)",
                   title_font=dict(size=13), gridcolor='rgba(200,200,200,0.3)',
                   showline=True, linecolor='#888', mirror=True),
        plot_bgcolor='#f8f9fb', paper_bgcolor='white',
        hovermode='closest',
        hoverlabel=dict(bgcolor='white', bordercolor='#ccc', font_size=12),
        width=1050, height=650,
        legend=dict(x=1.02, y=1, xanchor='left',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#ddd', borderwidth=1, font=dict(size=11)),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.0, y=1.17, xanchor="left",
            pad={"r": 6, "t": 6},
            bgcolor='#eef2f7', bordercolor='#ccd',
            font=dict(size=11), buttons=buttons,
            showactive=True, active=0,
        )],
        margin=dict(t=140, r=220, b=60, l=70),
    )
    fig.write_html(output_html)

def _plot_importance(all_studies: dict, output_html: str, output_png: str):
    from optuna.importance import get_param_importances

    param_names = ['x_close', 'ratio', 'lock_dens']
    accum = {p: [[], []] for p in param_names}

    for study in all_studies.values():
        for metric_idx in range(2):
            try:
                imp = get_param_importances(
                    study,
                    target=lambda t, i=metric_idx: t.values[i],
                )
                for p in param_names:
                    accum[p][metric_idx].append(imp.get(p, 0.0))
            except Exception:
                pass

    means = {
        p: [
            float(np.mean(accum[p][0])) if accum[p][0] else 0.0,
            float(np.mean(accum[p][1])) if accum[p][1] else 0.0,
        ]
        for p in param_names
    }

    labels  = ['x_close\n(порог закрытия)', 'ratio\n(отношение порогов)', 'lock_dens\n(плотность)']
    m1_vals = [means[p][0] for p in param_names]
    m2_vals = [means[p][1] for p in param_names]

    # PNG
    x     = np.arange(len(param_names))
    width = 0.35
    fig_mpl, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, m1_vals, width, label='M1 (здоровье)', color='#4C9BE8', alpha=0.85)
    bars2 = ax.bar(x + width/2, m2_vals, width, label='M2 (экономика)',  color='#E8714C', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Важность параметра', fontsize=12)
    ax.set_title('Важность параметров локдауна\n(усреднено по всем сценариям вакцины)', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()

    # HTML
    fig_px = go.Figure()
    fig_px.add_trace(go.Bar(
        name='M1 (здоровье)', x=param_names, y=m1_vals,
        marker_color='#4C9BE8', marker_line_color='white', marker_line_width=1.5,
        opacity=0.88, text=[f'{v:.2f}' for v in m1_vals], textposition='outside',
    ))
    fig_px.add_trace(go.Bar(
        name='M2 (локдаун)', x=param_names, y=m2_vals,
        marker_color='#E8714C', marker_line_color='white', marker_line_width=1.5,
        opacity=0.88, text=[f'{v:.2f}' for v in m2_vals], textposition='outside',
    ))
    fig_px.update_layout(
        barmode='group',
        title=dict(
            text='<b>Важность параметров локдауна</b><br>'
                 '<sup>усреднено по всем сценариям вакцины</sup>',
            font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center',
        ),
        xaxis=dict(
            ticktext=['x_close<br>(порог закрытия)', 'ratio<br>(отношение порогов)',
                      'lock_dens<br>(плотность)'],
            tickvals=param_names, tickfont=dict(size=12),
        ),
        yaxis=dict(title='Важность параметра', range=[0, 1.2],
                   gridcolor='rgba(200,200,200,0.3)'),
        plot_bgcolor='#f8f9fb', paper_bgcolor='white',
        legend=dict(font=dict(size=12)),
        width=700, height=480,
        margin=dict(t=100, b=80),
    )
    fig_px.write_html(output_html)