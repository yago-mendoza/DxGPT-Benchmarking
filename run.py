import pandas as pd

# Read the CSV files
before_df = pd.read_csv('before.csv')
after_df = pd.read_csv('after.csv')

# Create new columns in after_df
after_df['original_diagnosis'] = before_df['diagnosis']
after_df['icd10_diagnosis'] = before_df['icd10_diagnosis']

# Reorder columns to match desired structure
after_df = after_df[['id', 'case', 'diagnosis', 'original_diagnosis', 'icd10_diagnosis', 
                     'diagnostic_code', 'death', 'critical', 'pediatric', 'severity']]

# Save the modified dataframe
after_df.to_csv('after_.csv', index=False)
