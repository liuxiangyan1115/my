import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

train = pd.read_csv(r'C:\Users\64188\Desktop\loan_predict\train.csv')
test = pd.read_csv(r'C:\Users\64188\Desktop\loan_predict\test.csv')
submit = pd.read_csv(r'C:\Users\64188\Desktop\loan_predict\submit.csv')

cate_2_cols = ['XINGBIE', 'ZHIWU', 'XUELI']
cate_cols = ['HYZK', 'ZHIYE', 'ZHICHEN', 'DWJJLX', 'DWSSHY', 'GRZHZT']
num_cols = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE', 'DKFFE', 'DKYE', 'DKLL']

df = pd.concat([train, test], axis=0).reset_index(drop=True)
df['missing_rate'] = (df.shape[1] - df.count(axis=1)) / df.shape[1]

df['DKFFE_DKYE'] = df['DKFFE'] + df['DKYE']  # 贷款发放额+贷款余额
df['DKYE_DKFFE_ratio'] = df['DKYE'] / df['DKFFE_DKYE']
df['DKFFE_DKYE_ratio'] = df['DKFFE'] / df['DKFFE_DKYE']

df['DKFFE_DKY_multi_DKLL'] = (df['DKFFE'] + df['DKYE']) * df['DKLL']  # (贷款发放额+贷款余额)*贷款利率
df['DKFFE_multi_DKLL'] = df['DKFFE'] * df['DKLL']
df['DKYE_multi_DKLL'] = df['DKYE'] * df['DKLL']
df['DKFFE_multi_DKLL_ratio'] = df['DKFFE'] * df['DKLL'] / df['DKFFE_DKY_multi_DKLL']
df['DKYE_multi_DKLL_ratio'] = df['DKYE'] * df['DKLL'] / df['DKFFE_DKY_multi_DKLL']

df['GRYJCE_DWYJCE'] = df['GRYJCE'] + df['DWYJCE']  # 个人月缴存额+单位月缴存额
df['GRYJCE_DWYJCE_ratio'] = df['GRYJCE'] / df['GRYJCE_DWYJCE']
df['DWYJCE_GRYJCE_ratio'] = df['DWYJCE'] / df['GRYJCE_DWYJCE']

df['GRZHDNGJYE_GRZHSNJZYE'] = df['GRZHDNGJYE'] + df['GRZHSNJZYE']  # 个人账户当年归集余额+个人账户上年结转余额
df['GRZHYE_diff_GRZHDNGJYE'] = df['GRZHYE'] - df['GRZHDNGJYE']  # 个人账户余额-个人账户当年归集余额
df['GRZHYE_diff_GRZHSNJZYE'] = df['GRZHYE'] - df['GRZHSNJZYE']  # 个人账户余额-个人账户上年结转余额

gen_feats = ['DKFFE_DKYE', 'DKFFE_DKY_multi_DKLL', 'DKFFE_multi_DKLL', 'DKYE_multi_DKLL', 'GRYJCE_DWYJCE',
             'GRZHDNGJYE_GRZHSNJZYE', 'DKFFE_multi_DKLL_ratio', 'DKYE_multi_DKLL_ratio', 'GRZHYE_diff_GRZHDNGJYE',
             'GRZHYE_diff_GRZHSNJZYE', 'GRYJCE_DWYJCE_ratio', 'DWYJCE_GRYJCE_ratio', 'DKYE_DKFFE_ratio',
             'DKFFE_DKYE_ratio']


def get_age(df, col='age'):
    df[col + "_genFeat1"] = (df['age'] > 18).astype(int)
    df[col + "_genFeat2"] = (df['age'] > 25).astype(int)
    df[col + "_genFeat3"] = (df['age'] > 30).astype(int)
    df[col + "_genFeat4"] = (df['age'] > 35).astype(int)
    df[col + "_genFeat5"] = (df['age'] > 40).astype(int)
    df[col + "_genFeat6"] = (df['age'] > 45).astype(int)
    col_genFeat = []
    for i in range(1, 7):
        col_genFeat.append(col + '_genFeat{}'.format(i))
    return df, col_genFeat


df['age'] = ((1609430399 - df['CSNY']) / (365 * 24 * 3600)).astype(int)
df, genFeats1 = get_age(df, col='age')


# sns.distplot(df['age'][df['age'] > 0])
# plt.show()


def get_daikuanYE(df, col):
    df[col + '_genFeat1'] = (df[col] > 100000).astype(int)
    df[col + '_genFeat2'] = (df[col] > 120000).astype(int)
    df[col + '_genFeat3'] = (df[col] > 140000).astype(int)
    df[col + '_genFeat4'] = (df[col] > 180000).astype(int)
    df[col + '_genFeat5'] = (df[col] > 220000).astype(int)
    df[col + '_genFeat6'] = (df[col] > 260000).astype(int)
    df[col + '_genFeat7'] = (df[col] > 300000).astype(int)
    col_genFeat = []
    for i in range(1, 8):
        col_genFeat.append(col + '_genFeat{}'.format(i))
    return df, col_genFeat


df, genFeats2 = get_daikuanYE(df, col='DKYE')  # 贷款余额
df, genFeats3 = get_daikuanYE(df, col='DKFFE')  # 贷款发放额

plt.figure(figsize=(8, 2))
plt.subplot(1, 2, 1)
sns.distplot(df['DKYE'][df['label'] == 1])
plt.subplot(1, 2, 2)
sns.distplot(df['DKFFE'][df['label'] == 1])

for f in tqdm(cate_cols):
    df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique()))))
    df[f + '_count'] = df[f].map(df[f].value_counts())
    df = pd.concat([df, pd.get_dummies(df[f], prefix="{}".format(f))], axis=1)

cate_cols_combine = [[cate_cols[i], cate_cols[j]] for i in range(len(cate_cols)) \
                     for j in range(i + 1, len(cate_cols))]

for f1, f2 in tqdm(cate_cols_combine):
    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['id'].transform('count')
    df['{}_in_{}_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / df[f2 + '_count']
    df['{}_in_{}_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / df[f1 + '_count']

for f1 in tqdm(cate_cols):
    g = df.groupby(f1)
    for f2 in num_cols + gen_feats:
        for stat in ['sum', 'mean', 'std', 'max', 'min', 'std']:
            df['{}_{}_{}'.format(f1, f2, stat)] = g[f2].transform(stat)
    for f3 in genFeats2 + genFeats3:
        for stat in ['sum', 'mean']:
            df['{}_{}_{}'.format(f1, f2, stat)] = g[f2].transform(stat)

num_cols_gen_feats = num_cols + gen_feats
for f1 in tqdm(num_cols_gen_feats):
    g = df.groupby(f1)
    for f2 in num_cols_gen_feats:
        if f1 != f2:
            for stat in ['sum', 'mean', 'std', 'max', 'min', 'std']:
                df['{}_{}_{}'.format(f1, f2, stat)] = g[f2].transform(stat)

for i in tqdm(range(len(num_cols_gen_feats))):
    for j in range(i + 1, len(num_cols_gen_feats)):
        df['numsOf_{}_{}_add'.format(num_cols_gen_feats[i], num_cols_gen_feats[j])] = df[num_cols_gen_feats[i]] + df[
            num_cols_gen_feats[j]]
        df['numsOf_{}_{}_diff'.format(num_cols_gen_feats[i], num_cols_gen_feats[j])] = df[num_cols_gen_feats[i]] - df[
            num_cols_gen_feats[j]]
        df['numsOf_{}_{}_multi'.format(num_cols_gen_feats[i], num_cols_gen_feats[j])] = df[num_cols_gen_feats[i]] * df[
            num_cols_gen_feats[j]]
        df['numsOf_{}_{}_div'.format(num_cols_gen_feats[i], num_cols_gen_feats[j])] = df[num_cols_gen_feats[i]] / (
                df[num_cols_gen_feats[j]] + 0.0000000001)

num_cols = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE', 'DKFFE', 'DKYE']
df0 = df[num_cols]


def func_GRJCJS(x):
    if x <= 2402.75:
        return 1
    elif x > 2402.75 and x <= 3934.733:
        return 2
    elif x > 3934.733 and x <= 5877:
        return 3
    else:
        return 4


def func_GRZHYE(x):
    if x <= 1920.16375:
        return 1
    elif x > 1920.16375 and x <= 8529.122:
        return 2
    elif x > 8529.122 and x <= 19782.025:
        return 3
    else:
        return 4


def func_GRZHSNJZYE(x):
    if x <= 3007.135:
        return 1
    elif x > 3007.135 and x <= 9060.0775:
        return 2
    elif x > 9060.0775 and x <= 19698.5825:
        return 3
    else:
        return 4


def func_GRZHDNGJYE(x):
    if x <= -546.63375:
        return 1
    elif x > -546.63375 and x <= 771.8:
        return 2
    elif x > 771.8 and x <= 2492.04:
        return 3
    else:
        return 4


def func_GRYJCE(x):
    if x <= 412.5:
        return 1
    elif x > 412.5 and x <= 623.66:
        return 2
    elif x > 623.66 and x <= 858.315:
        return 3
    else:
        return 4


def func_DWYJCE(x):
    if x <= 412.5:
        return 1
    elif x > 412.5 and x <= 623.66:
        return 2
    elif x > 623.66 and x <= 858.315:
        return 3
    else:
        return 4


def func_DKFFE(x):
    if x <= 150237:
        return 1
    elif x > 150237 and x <= 250237:
        return 2
    else:
        return 4


def func_DKYE(x):
    if x <= 109612.025:
        return 1
    elif x > 109612.025 and x <= 146210.88:
        return 2
    elif x > 146210.88 and x <= 202249.99:
        return 3
    else:
        return 4


df0['GRJCJS_CLASS'] = df0['GRJCJS'].apply(func_GRJCJS)
df0['GRZHYE_CLASS'] = df0['GRZHYE'].apply(func_GRZHYE)
df0['GRZHSNJZYE_CLASS'] = df0['GRZHSNJZYE'].apply(func_GRZHSNJZYE)
df0['GRZHDNGJYE_CLASS'] = df0['GRZHDNGJYE'].apply(func_GRZHDNGJYE)
df0['GRYJCE_CLASS'] = df0['GRYJCE'].apply(func_GRYJCE)
df0['DWYJCE_CLASS'] = df0['DWYJCE'].apply(func_DWYJCE)
df0['DKFFE_CLASS'] = df0['DKFFE'].apply(func_DKFFE)
df0['DKYE_CLASS'] = df0['DKYE'].apply(func_DKYE)

num_cols = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE', 'DKFFE', 'DKYE']
for column in num_cols:
    groupby_list = [['GRJCJS_CLASS'], ['GRZHYE_CLASS'], ['GRZHSNJZYE_CLASS'], ['GRZHDNGJYE_CLASS'], ['GRYJCE_CLASS'],
                    ['DWYJCE_CLASS'], ['DKFFE_CLASS'], ['DKYE_CLASS']]
    for groupby in groupby_list:
        if 'GRJCJS_CLASS' in groupby and column == 'GRJCJS':
            continue
        if 'GRZHYE_CLASS' in groupby and column == 'GRZHYE':
            continue
        if 'GRZHSNJZYE_CLASS' in groupby and column == 'GRZHSNJZYE':
            continue
        if 'GRZHDNGJYE_CLASS' in groupby and column == 'GRZHDNGJYE':
            continue
        if 'GRYJCE_CLASS' in groupby and column == 'GRYJCE':
            continue
        if 'DWYJCE_CLASS' in groupby and column == 'DWYJCE':
            continue
        if 'DKFFE_CLASS' in groupby and column == 'DKFFE':
            continue
        if 'DKYE_CLASS' in groupby and column == 'DKYE':
            continue
        groupby_keylist = []
        for key in groupby:
            groupby_keylist.append(df0[key])
        tmp = df0[column].groupby(groupby_keylist).agg([sum, min, max, np.mean]).reset_index()
        tmp = pd.merge(df0, tmp, on=groupby, how='left')
        df0['ent_' + column.lower() + '-mean_gb_' + '_'.join(groupby).lower()] = df0[column] - tmp['mean']
        df0['ent_' + column.lower() + '-min_gb_' + '_'.join(groupby).lower()] = df0[column] - tmp['min']
        df0['ent_' + column.lower() + '-max_gb_' + '_'.join(groupby).lower()] = df0[column] - tmp['max']
        df0['ent_' + column.lower() + '/sum_gb_' + '_'.join(groupby).lower()] = df0[column] / tmp['sum']
df0.drop(['GRJCJS_CLASS', 'GRZHYE_CLASS', 'GRZHSNJZYE_CLASS', 'GRZHDNGJYE_CLASS', 'GRYJCE_CLASS', 'DWYJCE_CLASS',
          'DKFFE_CLASS', 'DKYE_CLASS'], axis=1, inplace=True)

cols1 = [col for col in df0.columns if col not in num_cols]

df = pd.concat([df, df0[cols1]], axis=1)

# 训练集测试集
label_df = df['label']
df.dropna(axis=1, inplace=True)
df = pd.concat([df, label_df], axis=1)

train_df = df[df['label'].isna() == False].reset_index(drop=True)
test_df = df[df['label'].isna() == True].reset_index(drop=True)
print(train_df.shape, test_df.shape)

drop_feats = [f for f in train_df.columns if train_df[f].nunique() == 1 or train_df[f].nunique() == 0]
cols = [col for col in train_df.columns if col not in ['id', 'label'] + drop_feats]

# 平衡样本
n_sample = train_df.shape[0]
pos_n_sample = train_df[train_df['label'] == 0].shape[0]
neg_n_sample = train_df[train_df['label'] == 1].shape[0]
print('样本个数：{}；正样本占{:.2%}；负样本占{:.2%}'.format(n_sample, pos_n_sample / n_sample, neg_n_sample / n_sample))

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
x, y = sm.fit_sample(train_df[cols], train_df['label'])
print('通过SMOTE方法平衡正负样本后')
n_sample = y.shape[0]
pos_n_sample = y[y == 0].shape[0]
neg_n_sample = y[y == 1].shape[0]
print('样本个数：{}；正样本占{:.2%}；负样本占{:.2%}'.format(n_sample, pos_n_sample / n_sample, neg_n_sample / n_sample))

train_df = pd.concat([x, y], axis=1)

from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_auc_score
import time

oof = np.zeros(train_df.shape[0])
test_df['prob'] = 0
clf = LGBMClassifier(
    learning_rate=0.05,
    n_estimators=10230,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=1023,
    metric=None
)

val_aucs = []
seeds = [1023, 2048, 2098]
for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print('--------------------- {} fold ---------------------'.format(i))
        t = time.time()
        trn_x, trn_y = train_df[cols].iloc[trn_idx].reset_index(drop=True), train_df['label'].values[trn_idx]
        val_x, val_y = train_df[cols].iloc[val_idx].reset_index(drop=True), train_df['label'].values[val_idx]
        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)],
                eval_metric='auc',
                early_stopping_rounds=200,
                verbose=200)
        oof[val_idx] = clf.predict_proba(val_x)[:, 1]
        test_df['prob'] += clf.predict_proba(test_df[cols])[:, 1] / skf.n_splits / len(seeds)

    cv_auc = roc_auc_score(train_df['label'], oof)
    val_aucs.append(cv_auc)
    print('\ncv_auc: ', cv_auc)
print(val_aucs, np.mean(val_aucs))


def tpr_weight_funtion(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]

    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3


tpr = round(tpr_weight_funtion(train_df['label'], oof), 6)
print(tpr, round(np.mean(val_aucs), 5))

submit['id'] = test_df['id']
submit['label'] = test_df['prob']

submit.to_csv(r'C:\Users\64188\Desktop\loan_predict\submission{}_{}.csv'.format(tpr, round(np.mean(val_aucs), 6)),
              index=False)
