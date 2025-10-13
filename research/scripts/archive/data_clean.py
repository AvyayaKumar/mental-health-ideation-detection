import pandas as pd

# import dataset
csv_path = "Suicide_Detection.csv"
df = pd.read_csv(csv_path)

# create dictionary map
label_map = {'non-suicide': 0, 'suicide': 1}

# map the data
df['class'] = df['class'].map(label_map)

# print class counts
print(df['class'].value_counts())

# save the new dataset
df.to_csv("mapped_dataset.csv", index=False)
