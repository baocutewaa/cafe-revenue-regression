import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv', parse_dates=['date'])
test = pd.read_csv('data/test.csv', parse_dates=['date'])
stores = pd.read_csv('data/stores.csv')
weather = pd.read_csv('data/weather.csv', parse_dates=['date'])
holidays = pd.read_csv('data/holidays.csv', parse_dates=['date'])

print('--- CHECKING MISSING DATES IN TEST ---')
full_test_dates = pd.date_range('2026-01-01', '2026-03-31')
print('Total days in Q1 2026:', len(full_test_dates))
print('Unique counts of days per store in test:')
print(test.groupby('store_id')['date'].count().value_counts())

for sid, grp in test.groupby('store_id'):
    missing_days = set(full_test_dates) - set(grp['date'])
    if len(missing_days) > 0:
        st = stores[stores['store_id'] == sid]['store_type'].values[0]
        dates_sorted = sorted(list(missing_days))
        formatted = [d.strftime('%Y-%m-%d') for d in dates_sorted]
        print(f'Store {sid} ({st:12s}) missing {len(missing_days)} days: {formatted}')

print('\n--- CHECKING TET DATES IN TRAIN 2024 & 2025 ---')
# In train 2024 and 2025, which stores were missing during Tet?
for yr, m1_str in [(2024, '2024-02-10'), (2025, '2025-01-29')]:
    m1 = pd.Timestamp(m1_str)
    sub_tr = train[(train['date'] >= m1 - pd.Timedelta(days=5)) & (train['date'] <= m1 + pd.Timedelta(days=7))]
    print(f'Year {yr} around Tet ({m1_str}): total stores present = {sub_tr["store_id"].nunique()}')
    # Which stores are missing on Mung 1?
    m1_stores = set(sub_tr[sub_tr['date'] == m1]['store_id'])
    all_stores_yr = set(train[train['date'].dt.year == yr]['store_id'])
    closed_m1 = all_stores_yr - m1_stores
    print(f'Stores closed on Mung 1 ({m1_str}): {len(closed_m1)} stores: {sorted(list(closed_m1))}')
