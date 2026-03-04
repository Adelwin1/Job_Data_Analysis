import numpy as np
import pandas as pd
from streamlit import columns

df = pd.read_csv("../data/salary_survey_raw.csv")
print(df.head())
print("Shape of df: ", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\n Data Types:\n", df.dtypes)
print("\n Missing % by columns:\n")
print((df.isna().mean()*100).sort_values(ascending=False))
print("\n Duplicate rows:\n")
print(df.duplicated().sum())

df.rename(columns={"What is your annual salary? (You'll indicate the currency in a later question. If you are part-time or hourly, please enter an annualized equivalent -- what you would earn if you worked the job 40 hours a week, 52 weeks a year.)":
"annual_salary"}, inplace=True)
df['annual_salary_clean'] = (df['annual_salary'].astype(str)
                             .str.replace(r'[^\d.]', ''
                             , regex=True).replace('', np.nan)
                             .astype (float)
                    )
print("\n Cleaned Salary Columns:")
print(df[['annual_salary', 'annual_salary_clean']].head())
print("\n Missing Salaries after the cleaning total:", df['annual_salary_clean'].isna().sum())

df.rename(columns={
    "How many years of professional work experience do you have overall?": "years_experience_total",
    "How many years of professional work experience do you have in your field?": "years_experience_field"
}, inplace=True)

def clean_experience (x):
    if pd.isna(x):
        return np.nan
    x= str(x).strip()

    if x.startswith('<'):
        return 0.5
    if x.endswith('+'):
        return float(x[:-1])
    try:
        return float(x)
    except:
        return np.nan


df['years_experience_total'] = df['years_experience_total'].apply(clean_experience)
df['years_experience_field'] = df['years_experience_field'].apply(clean_experience)

print("\n Cleaned Experience Columns:")
print(df[['years_experience_total', 'years_experience_field']].head(10))

text_cols = ['Job title', 'What industry do you work in?', 'What country do you work in?', 'What city do you work in?']

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
df['What country do you work in?'] = df['What country do you work in?'].replace({
    'usa':'united states',
    'us':'united states',
     'u.s':'united states',}
)
print("\n Sample free-text normalization:")
pd.set_option('display.max_columns', None)

print(df[text_cols].head())
high_missing_cols = ['If "Other," please indicate the currency here:',
    'If your income needs additional context do you provide it here:',
    'If your job title needs additional context, please clarify here:']
df.drop(columns=high_missing_cols, inplace=True, errors='ignore')
print("\n Sample after dropping all high-missing ones:")
print(df.columns.tolist())

df.rename(columns={'How much additional monetary compensation do you get, if any (for example, bonuses or overtime in an average year)? Please only include monetary compensation here, not the value of benefits.': 'bonus_col'}, inplace=True)

df['total_compensation'] = df['annual_salary_clean']+ df['bonus_col'].fillna(0)
print("\n Sample  total compensation:")
print(df[['annual_salary_clean', 'bonus_col','total_compensation' ]].head(5))


df['experience_ratio'] = df['years_experience_field']/df['years_experience_total']
df['experience_ratio'].replace([np.inf, -np.inf], np.nan)

print("\n Sample  experience ratio:")
print(df[['years_experience_total', 'years_experience_field', 'experience_ratio']].head(5))

def seniority(exp):
    if pd.isna(exp):
        return 'unknown'
    elif exp < 3:
        return 'junior'
    elif exp < 7:
        return 'mid'
    else:
        return 'senior'

df['seniority_level'] = df['years_experience_total'].apply(seniority)

print("\nSample seniority levels:")
print(df[['years_experience_total', 'seniority_level']].head(10))
initial_rows = df.shape[0]
df = df.drop_duplicates()
final_rows = df.shape[0]
print(final_rows - initial_rows)

df.columns = df.columns.str.strip().str.lower().str.replace(r'[^\w+]', '_', regex=True)
print(df.columns)

numeric_cols = ["annual_salary_clean", "bonus_col", "total_compensation", "years_experience_total", "years_experience_field"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(df.dtypes)
print(df.head(5))

[col for col in df.columns if "timestamp" in col]
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
print(df["timestamp"])
print(df.columns)
print(df["please_indicate_the_currency"].value_counts())
df.rename(columns ={'please_indicate_the_currency' : 'currency' }, inplace=True)
fx_rates = {
    "usd": 1.0,
    "cad": 0.75,
    "gbp": 1.25,
    "eur": 1.08,
    "aud": 0.65
}
df["currency"] = df["currency"].str.lower().str.strip()
df["fix_rates"] = df["currency"].map(fx_rates)
df["salary_usd"]= df["annual_salary_clean"]* df["fix_rates"]
print(df["salary_usd"].head(3))
print(df[df["fix_rates"].isna()]["currency"].unique())
kofi = (df["salary_usd"]<=0).sum()
print(kofi)
df = df[df["salary_usd"]>0]
df["log_salary_usd"] = np.log(df["salary_usd"])
print(df["log_salary_usd"].head(3))

df["country_standardized"] = df["what_country_do_you_work_in_"]
df["industry_standardized"] = df["what_industry_do_you_work_in_"]

country_map = {
    "united states of america": "united states",
    "u.s.a.": "united states"
}

df["country_standardized"] = df["country_standardized"].replace(country_map)
df["country_standardized"] = df["country_standardized"].str.title()
df["industry_standardized"] = df["industry_standardized"].str.title()

print(df["country_standardized"].head(3))
df["salary_missing_flag"] = df["salary_usd"].isna().astype(int)

df["salary_usd"] = df["salary_usd"].fillana(df["salary_usd"].median())
df["experience_ratio"] = df["experience_ratio"].replace([np.inf, -np.inf], np.nan)