import pandas as pd
import numpy as np
from scipy import stats
from numpy.random import default_rng

# ===================== 配置 =====================
FILE_PATH = r"D:\建模国赛\附件_filtered_merged.xlsx"  # ←← 修改为你的文件路径
SHEET_NAME = 0
N = 2000  # 生成人数
RNG = default_rng(42)

# 变量列名（按你的文件）
COL_AGE = "年龄"
COL_BMI = "孕妇BMI"  # 若无此列、但有“BMI”会自动兜底
COL_BMI_ALT = "BMI"
COL_GRAV = "怀孕次数"
COL_PAR = "生产次数"

# 逻辑约束与裁剪（可按需调整）
AGE_MIN, AGE_MAX = 15, 55
BMI_MIN, BMI_MAX = 14, 45

# ===================== 工具函数 =====================
def to_numeric_clean(s: pd.Series) -> pd.Series:
    """去掉百分号等字符后转数值"""
    return pd.to_numeric(
        s.astype(str).str.replace('%','',regex=False).str.replace('％','',regex=False).str.strip(),
        errors='coerce'
    )

def fit_normal(x: np.ndarray):
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    ll = np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))
    aic = 2*2 - 2*ll
    return {"name":"normal","params":{"mu":mu,"sigma":sigma},"AIC":aic}

def fit_lognorm(x: np.ndarray):
    # 仅对正数有效
    x_pos = x[x>0]
    if len(x_pos) < max(30, 0.5*len(x)):  # 太多非正值则放弃
        return {"name":"lognorm","params":None,"AIC":np.inf}
    # scipy 的 lognorm 参数：shape=s, loc, scale=exp(mu)
    s, loc, scale = stats.lognorm.fit(x_pos, floc=0)
    ll = np.sum(stats.lognorm.logpdf(x_pos, s, loc=loc, scale=scale))
    aic = 2*2 - 2*ll  # s, scale 两个自由参数（loc固定）
    return {"name":"lognorm","params":{"s":s,"loc":loc,"scale":scale},"AIC":aic}

def choose_continuous(x: np.ndarray):
    f1, f2 = fit_normal(x), fit_lognorm(x)
    return f1 if f1["AIC"] <= f2["AIC"] else f2

def fit_poisson(k: np.ndarray):
    lam = float(np.mean(k))
    ll = np.sum(stats.poisson.logpmf(k, lam))
    aic = 2*1 - 2*ll
    return {"name":"poisson","params":{"lam":lam},"AIC":aic}

def fit_nbinom(k: np.ndarray):
    m, v = float(np.mean(k)), float(np.var(k, ddof=1))
    if v <= m + 1e-12:  # 无过度离散，负二项退化
        return {"name":"nbinom","params":None,"AIC":np.inf}
    # 矩法估计：n, p
    n_param = m*m/(v - m)
    p_param = n_param/(n_param + m)
    ll = np.sum(stats.nbinom.logpmf(k, n_param, p_param))
    aic = 2*2 - 2*ll
    return {"name":"nbinom","params":{"n":n_param,"p":p_param},"AIC":aic}

def choose_count(k: np.ndarray):
    f1, f2 = fit_poisson(k), fit_nbinom(k)
    return f1 if f1["AIC"] <= f2["AIC"] else f2

def inv_sample_cont(u: np.ndarray, fit: dict) -> np.ndarray:
    if fit["name"] == "normal":
        mu, sigma = fit["params"]["mu"], fit["params"]["sigma"]
        return stats.norm.ppf(u, loc=mu, scale=sigma)
    else:
        s, loc, scale = fit["params"]["s"], fit["params"]["loc"], fit["params"]["scale"]
        return stats.lognorm.ppf(u, s, loc=loc, scale=scale)

def inv_sample_count(u: np.ndarray, fit: dict) -> np.ndarray:
    if fit["name"] == "poisson":
        lam = fit["params"]["lam"]
        return stats.poisson.ppf(u, mu=lam).astype(int)
    else:
        n_param, p_param = fit["params"]["n"], fit["params"]["p"]
        return stats.nbinom.ppf(u, n_param, p_param).astype(int)

# ===================== 读取/清洗 =====================
df_raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

# 兜底：若“孕妇BMI”缺失，用“BMI”
if COL_BMI not in df_raw.columns and COL_BMI_ALT in df_raw.columns:
    df_raw[COL_BMI] = df_raw[COL_BMI_ALT]

need_cols = [COL_AGE, COL_BMI, COL_GRAV, COL_PAR]
missing = [c for c in need_cols if c not in df_raw.columns]
if missing:
    raise KeyError(f"缺少必要列：{missing}；请检查文件列名。")

df = pd.DataFrame({
    "Age": to_numeric_clean(df_raw[COL_AGE]),
    "BMI": to_numeric_clean(df_raw[COL_BMI]),
    "Gravidity": df_raw[COL_GRAV].astype(str).str.strip(),
    "Parity": df_raw[COL_PAR].astype(str).str.strip(),
})

# 处理“>=3”等文本情况 → 先替换再转数值
df["Gravidity"] = df["Gravidity"].replace({">=3": "3", "≥3": "3"})
df["Parity"]    = df["Parity"].replace({">=3": "3", "≥3": "3"})

df["Gravidity"] = to_numeric_clean(df["Gravidity"]).round()
df["Parity"]    = to_numeric_clean(df["Parity"]).round()

# 丢弃缺失
df = df.dropna(subset=["Age","BMI","Gravidity","Parity"]).copy()

# 合理化：次数非负整数；且 Parity ≤ Gravidity
df["Gravidity"] = np.clip(df["Gravidity"].astype(int), 0, None)
df["Parity"]    = np.clip(df["Parity"].astype(int), 0, None)
df["Parity"]    = np.minimum(df["Parity"], df["Gravidity"])

# ===================== 拟合边际分布（独立） =====================
age_fit = choose_continuous(df["Age"].values)
bmi_fit = choose_continuous(df["BMI"].values)
grav_fit = choose_count(df["Gravidity"].values)
par_fit  = choose_count(df["Parity"].values)

print("边际分布拟合（独立采样将使用下列分布）：")
for name, fit in [("Age", age_fit), ("BMI", bmi_fit), ("Gravidity", grav_fit), ("Parity", par_fit)]:
    print(f"  {name}: {fit['name']}  params={fit['params']}  AIC={fit['AIC']:.3f}")

# ===================== 独立采样生成 =====================
u_age  = RNG.random(N)
u_bmi  = RNG.random(N)
u_g    = RNG.random(N)
u_p    = RNG.random(N)

Age_syn = inv_sample_cont(u_age, age_fit)
BMI_syn = inv_sample_cont(u_bmi, bmi_fit)
Grav_syn = inv_sample_count(u_g, grav_fit)
Par_syn  = inv_sample_count(u_p, par_fit)

# 逻辑约束与裁剪
Age_syn = np.clip(Age_syn, AGE_MIN, AGE_MAX)
BMI_syn = np.clip(BMI_syn, BMI_MIN, BMI_MAX)
Grav_syn = np.clip(Grav_syn, 0, None)
Par_syn  = np.clip(Par_syn, 0, None)
Par_syn  = np.minimum(Par_syn, Grav_syn)

# 输出表
syn = pd.DataFrame({
    "Age": np.round(Age_syn, 2),
    "BMI": np.round(BMI_syn, 2),
    "Gravidity": Grav_syn.astype(int),
    "Parity": Par_syn.astype(int),
})

# 保存
out_xlsx = "synthetic_independent_2000.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    syn.to_excel(writer, index=False, sheet_name="Synthetic_Independent")
    # 同时保存拟合参数，便于复现
    fit_tab = pd.DataFrame([
        {"Variable":"Age", "Dist":age_fit["name"], "Params":str(age_fit["params"]), "AIC":age_fit["AIC"]},
        {"Variable":"BMI", "Dist":bmi_fit["name"], "Params":str(bmi_fit["params"]), "AIC":bmi_fit["AIC"]},
        {"Variable":"Gravidity", "Dist":grav_fit["name"], "Params":str(grav_fit["params"]), "AIC":grav_fit["AIC"]},
        {"Variable":"Parity", "Dist":par_fit["name"], "Params":str(par_fit["params"]), "AIC":par_fit["AIC"]},
    ])
    fit_tab.to_excel(writer, index=False, sheet_name="Marginal_Fits")

print(f"已生成独立采样的 2000 人样本：{out_xlsx}")
