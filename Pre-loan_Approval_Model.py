import pandas as pd
import numpy as np
import toad
useful_cols = [
    'loan_amnt',        # 贷款金额
    'term',             # 期限
    'int_rate',         # 利率（注意：如果是预测违约，建议剔除；如果是模拟审批，可保留）
    'installment',      # 月供
    'grade',            # 内部评级
    'sub_grade',        # 内部子评级
    'emp_length',       # 工作年限
    'home_ownership',   # 房屋所有权
    'annual_inc',       # 年收入
    'verification_status', # 收入验证状态
    'purpose',          # 贷款目的
    'dti',              # 负债收入比
    'delinq_2yrs',      # 2年内逾期次数
    'fico_range_low',   # FICO最低分
    'fico_range_high',  # FICO最高分
    'inq_last_6mths',   # 6个月内查询次数
    'open_acc',         # 信用账户数
    'pub_rec',          # 公共违约记录数
    'revol_bal',        # 循环信用余额
    'revol_util',       # 循环信用利用率
    'total_acc',        # 总账户数
    'loan_status'       # 贷款状态（目标变量）
]
df = pd.read_csv('data/accepted_2007_to_2018Q4.csv.gz', usecols=useful_cols, low_memory=False)
print(df.head())

print("原始数据总行数:", df.shape[0])
# 1. 查看 loan_status 的分布情况
print("\n贷款状态分布：")
print(df['loan_status'].value_counts())
# 2. 过滤，只保留 'Fully Paid' 和 'Charged Off'
valid_status = ['Fully Paid', 'Charged Off']
df = df[df['loan_status'].isin(valid_status)].copy()
# 3. 创建目标变量 target: 1=违约(坏), 0=正常(好)
df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
# 4. 把 loan_status 列删掉，避免信息泄漏（因为已经提取出标签了）
df = df.drop(columns=['loan_status'])
print("\n清洗后数据总行数:", df.shape[0])
print("违约率（坏客户占比）:")
print(df['target'].mean())
# 剔除有“数据穿越”风险的字段（审批后才知道的信息）
leakage_cols = ['int_rate', 'grade', 'sub_grade', 'installment']
df = df.drop(columns=leakage_cols, errors='ignore')

# 再检查一下现在还有哪些列
print("清洗后剩余的特征列:")
print(df.columns.tolist())
print("\n现在数据大小:", df.shape)

print("=== 1. 检查缺失值分布 ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
print(missing_df[missing_df['缺失数量'] > 0].sort_values(by='缺失比例(%)', ascending=False))

print("\n=== 2. 清洗 term 字段 ===")
# '36 months' -> 36
# ===== 执行清洗 =====
df['term'] = df['term'].astype(str)
df['term'] = df['term'].str.strip()          # 去掉首尾空格
df['term'] = df['term'].str.lower()          # 全部转小写
df['term'] = df['term'].str.replace('months', '').astype(int)  # 去掉 " months" 并将字符串转换为整数类型
print("term 清洗后唯一值:", df['term'].unique())

print("\n=== 3. 清洗 emp_length 字段 ===")
# '10+ years' -> 10, '< 1 year' -> 0, 'n/a' -> np.nan
df['emp_length'] = df['emp_length'].astype(str)
df['emp_length'] = df['emp_length'].str.replace(' years', '').str.replace('+', '').str.replace('< 1 year', '0')  #有不严谨的地方
df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce') # 把'n/a'变成空值
print("emp_length 清洗后唯一值:", sorted(df['emp_length'].dropna().unique()))

print("\n=== 4. 处理 annual_inc 和 dti 的异常值 ===")
# 年收入超过99.9%分位数的，直接截断（Winsorize缩尾）
inc_99 = df['annual_inc'].quantile(0.999)
print(f"年收入 99.9% 分位数为: {inc_99:.2f}，将大于此值的记录截断")
df.loc[df['annual_inc'] > inc_99, 'annual_inc'] = inc_99
# 负债比 dti 也是同理，极端值截断
dti_99 = df['dti'].quantile(0.999)
df.loc[df['dti'] > dti_99, 'dti'] = dti_99
print(f"负债比 dti 99.9% 分位数为: {dti_99:.2f}，已截断")

print("\n=== 5. 缺失值填充 ===")
# emp_length 缺失，通常用众数（最常见的工作年限）填充，或者直接用中位数
df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())#直接用中位数能满足要求吗？

# revol_util（信用利用率）缺失，通常用0或中位数填充
# 1. 先把字符串清洗成数字
if 'revol_util' in df.columns:
    df['revol_util'] = df['revol_util'].astype(str)
    df['revol_util'] = df['revol_util'].str.replace('%', '', regex=False)
    df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')  # 'n/a' 变 NaN
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())

# 其他可能存在的缺失值（如 dti 少量缺失），统一用中位数填充
df = df.fillna(df.median(numeric_only=True))

print(f"\n清洗完成！当前数据大小: {df.shape}")
print("当前还有多少缺失值:", df.isnull().sum().sum())

print("=== 开始特征工程 ===")

# 1. 贷款收入比（反映还款压力）
# 贷款金额占年收入的比例，越高代表负债越重
df['loan_inc_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)

# 2. 信用利用率（反映资金饥渴度）
# 已经有 revol_util 了，但我们取个对数平滑一下极端值
df['revol_util_log'] = np.log1p(df['revol_util'])

# 3. 信用历史长度（间接反映稳定性）
# 用总账户数/信用历史长度，数值越大说明信用活动越频繁  '这里加入进一步分析，（还款质量）进一步说明是不是能构造一个新特征'
df['credit_activity_ratio'] = df['total_acc'] / (df['open_acc'] + 1)

# 4. 处理无穷值（除以0可能产生Inf）
df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))

print(f"特征工程完成，当前特征数量: {df.shape[1]}")
print("新增特征预览:")
print(df[['loan_inc_ratio', 'revol_util_log', 'credit_activity_ratio']].head())

# ==================== 第四步：采样 ====================
# 从134万条中随机采样20万条，内存占用小，跑得快
df_sample = df.sample(n=200000, random_state=42).reset_index(drop=True)
print(f"\n采样完成,采样数据量: {df_sample.shape}，违约率: {df_sample['target'].mean():.2%}")

# ==================== 第五步：分箱与IV计算（修正版） ====================
print("\n开始卡方分箱")
X = df_sample.drop(columns=['target'])
y = df_sample['target']

# 分箱
combiner = toad.transform.Combiner()
combiner.fit(X, y, method='chi', min_samples=0.10, n_bins=4)

# WOE转换
X_woe = combiner.transform(X)
woe_transformer = toad.transform.WOETransformer()
X_woe = woe_transformer.fit_transform(X_woe, y)

# 计算IV（修正：使用 toad.quality 一键计算所有特征的IV）
iv_df = toad.quality(df_sample, 'target', indicators=['iv'])
iv_df = iv_df[['iv']].sort_values(by='iv', ascending=False).reset_index()
iv_df.columns = ['特征', 'IV值']

print("\n=== 所有特征的 IV 值排行榜 ===")
print(iv_df.to_string(index=False))

#算法是冰冷的，它无法完全替代业务逻辑。 如果完全盲信 toad 给出的 IV 排行榜，就会直接错失 FICO 这个风控领域的好工具，所以要结合业务逻辑去利用更多的方法找到真正能够建立模型的变量或者说特征值。

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd

print("=== 开始模型训练 ===")

# 1. 根据 IV 排行榜，筛选出 IV >= 0.02 的特征
selected_features = ['annual_inc', 'home_ownership', 'inq_last_6mths',
                     'revol_util_log', 'credit_activity_ratio', 'purpose']

# X_woe 是之前分箱并 WOE 转换后的数据，直接按列名取子集
X_selected = X_woe[selected_features]
y = df_sample['target']

# 2. 分层划分训练集和测试集（保证坏人比例一致）
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 3. 训练逻辑回归（class_weight='balanced' 自动处理样本不平衡）
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 4. 在测试集上预测违约概率
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 5. 计算 AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n=== 模型评估 ===")
print(f"AUC: {auc:.4f}")

# 6. 手写计算 KS（sklearn 没有现成的 KS 函数）
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
ks = max(tpr - fpr)
print(f"KS: {ks:.4f}")

# 7. 查看模型系数（了解每个特征对风险的影响方向）
coef_df = pd.DataFrame({
    '特征': selected_features,
    '系数': model.coef_[0]
})
print("\n=== 模型系数 ===")
print(coef_df.to_string(index=False))
'''这里是发现FICO被原模型IV结果抛弃后的业务直觉，所以重新推敲了FICO的重要性'''
# 检查 FICO 分数在清洗后的 df_sample 中是否还有区分度
print("FICO分数在 df_sample 中的分布：")
print(df_sample[['fico_range_low', 'fico_range_high']].describe())

print("\nFICO分数与违约率的关系（按四分位分箱）：")
df_sample['fico_bin'] = pd.qcut(df_sample['fico_range_low'], q=4, duplicates='drop')
print(df_sample.groupby('fico_bin')['target'].mean())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

print("=== 开始训练原始特征逻辑回归模型 ===")

# 1. 选取核心特征（包含 FICO，以及之前IV筛选出的特征）
raw_features = [
    'loan_amnt', 'annual_inc', 'dti', 'delinq_2yrs',
    'fico_range_low',      # 核心强特征！
    'inq_last_6mths', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc',
    'loan_inc_ratio', 'credit_activity_ratio'
]

X = df_sample[raw_features].copy()
y = df_sample['target']

# 2. 标准化（逻辑回归必须做，因为 FICO 是几百，收入是几万，量纲差异大）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 分层划分（保证坏客户比例一致:'stratify=y'）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 4. 训练逻辑回归（应对样本不平衡）
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 5. 预测与评估
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ks = max(tpr - fpr)

print(f"\n=== 模型评估 ===")
print(f"AUC: {auc:.4f}")
print(f"KS: {ks:.4f}")

# 6. 查看模型系数，重点关注 FICO 的方向
coef_df = pd.DataFrame({
    '特征': raw_features,
    '系数': model.coef_[0]
})
coef_df['系数绝对值'] = coef_df['系数'].abs()
coef_df = coef_df.sort_values(by='系数绝对值', ascending=False).drop(columns='系数绝对值')

print("\n=== 模型系数（按影响大小排序）===")
print(coef_df.to_string(index=False))

'''这里是决策树阶段，也可以避免多重共线性的影响'''
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== 开始训练 XGBoost 模型 ===")

# 1. 选取特征（保持和逻辑回归完全一致，方便对比）
raw_features = [
    'loan_amnt', 'annual_inc', 'dti', 'delinq_2yrs',
    'fico_range_low', 'inq_last_6mths', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc',
    'loan_inc_ratio', 'credit_activity_ratio'
]

X_xgb = df_sample[raw_features]
y = df_sample['target']

# 2. 分层划分（树模型不需要标准化，直接传入原始数值）
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集大小: {X_train_xgb.shape}, 测试集大小: {X_test_xgb.shape}")

# 3. 计算 scale_pos_weight（解决样本极度不平衡）
# 负样本数 / 正样本数，告诉模型“这1个坏人，相当于4个好人那么重要”
scale_pos_weight = (y_train_xgb == 0).sum() / (y_train_xgb == 1).sum()
print(f"scale_pos_weight 计算值: {scale_pos_weight:.2f}")

# 4. 初始化 XGBoost 分类器
xgb_model = xgb.XGBClassifier(
    n_estimators=100,              # 种100棵树
    max_depth=4,                   # 每棵树深度4（防止过拟合）
    learning_rate=0.1,             # 学习率，每棵树只学10%
    scale_pos_weight=scale_pos_weight, # 处理不平衡
    random_state=42,
    eval_metric='auc'              # 评估指标用AUC
)

# 5. 训练模型
print("正在训练 XGBoost（预计1-2分钟）...")
xgb_model.fit(X_train_xgb, y_train_xgb)

# 6. 预测与评估
y_pred_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]

# 计算 AUC
auc_xgb = roc_auc_score(y_test_xgb, y_pred_xgb)

# 手写计算 KS
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_xgb, y_pred_xgb)
ks_xgb = max(tpr_xgb - fpr_xgb)

print(f"\n=== XGBoost 模型评估 ===")
print(f"AUC: {auc_xgb:.4f}")
print(f"KS: {ks_xgb:.4f}")

# 7. 提取特征重要性并可视化
importance_df = pd.DataFrame({
    '特征': raw_features,
    '重要性': xgb_model.feature_importances_
}).sort_values(by='重要性', ascending=False)

print("\n=== 特征重要性排行 ===")
print(importance_df.to_string(index=False))

# 画个水平条形图，这张图直接截图放进简历！
plt.figure(figsize=(10, 6))
plt.barh(importance_df['特征'], importance_df['重要性'], color='steelblue')
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance - Lending Club Default Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=300)
print("\n特征重要性图片已保存为 'xgb_feature_importance.png'")