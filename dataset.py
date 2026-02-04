import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 20000

real_news = [
    "Government announces new policy",
    "Scientists discover new planet",
    "Economy shows steady growth",
    "Education reforms approved",
    "Health ministry releases new guidelines"
]

fake_news = [
    "Aliens landed in city",
    "Celebrity replaced by clone",
    "Miracle cure discovered overnight",
    "Government hiding secret technology",
    "Time travel proven by YouTuber"
]

texts = []
labels = []

for i in range(n_samples):
    if i % 2 == 0:
        texts.append(np.random.choice(real_news))
        labels.append(0)   # REAL
    else:
        texts.append(np.random.choice(fake_news))
        labels.append(1)   # FAKE

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

df.to_csv("news_dataset_20000.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print(df['label'].value_counts())
