import pickle, glob, os
import numpy as np
import pandas as pd

TREES = os.path.expanduser('~/KPTracer-Data/trees')
rows_tumor, rows_site = [], []

for f in sorted(glob.glob(f'{TREES}/*_priors.pkl')):
    tumor = os.path.basename(f).replace('_priors.pkl', '')
    with open(f, 'rb') as fh:
        priors = pickle.load(fh)

    # observed distinct mutated states per site in the character matrix (for comparison)
    cm_path = f'{TREES}/{tumor}_character_matrix.txt'
    obs = {}
    if os.path.exists(cm_path):
        cm = pd.read_csv(cm_path, sep='\t', index_col=0)
        cm = cm.apply(pd.to_numeric, errors='coerce')
        for i, col in enumerate(cm.columns):
            v = cm[col].dropna()
            obs[i] = int(v[v > 0].nunique())

    n_prior = []
    for site in sorted(priors.keys()):
        n = len(priors[site])
        n_prior.append(n)
        rows_site.append({'tumor': tumor, 'site': site,
                          'n_states_priors': n,
                          'n_states_observed': obs.get(site, np.nan),
                          'prior_sum': float(sum(priors[site].values()))})
    n_prior = np.array(n_prior)
    n_obs = np.array([obs[s] for s in sorted(obs)]) if obs else np.array([np.nan])

    rows_tumor.append({'tumor': tumor, 'n_sites_priors': len(n_prior),
                       'n_sites_cm': len(obs),
                       'm_priors_min': n_prior.min(), 'm_priors_max': n_prior.max(),
                       'm_priors_mean': round(n_prior.mean(), 2),
                       'm_priors_median': float(np.median(n_prior)),
                       'm_obs_mean': round(np.nanmean(n_obs), 2)})

pd.DataFrame(rows_tumor).to_csv('m_per_tumor.csv', index=False)
pd.DataFrame(rows_site).to_csv('m_per_site.csv', index=False)
print(pd.DataFrame(rows_tumor).to_string(index=False))
print(f'\n{len(rows_tumor)} tumors written to m_per_tumor.csv / m_per_site.csv')
