import pandas as pd

# Read the CSV file with UTF-8 encoding
file_path = 'E:\\agent_dev\\langGraph\\student_habits_performance.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# Print the first 5 rows to confirm data structure
print(data.head())