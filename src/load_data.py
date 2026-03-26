import pandas as pd
from .preprocessing import clean_text

def load_and_prepare_data(fake_path="data/Fake.csv", true_path="data/True.csv"):
    # Load both datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add label column
    fake_df['label'] = 0  # Fake
    true_df['label'] = 1  # Real

    # Merge datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42)

    # Combine title + text
    df['text'] = df['title'] + " " + df['text']

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Keep only needed columns
    df = df[['text', 'label']]

    return df

if __name__ == "__main__":
    df = load_and_prepare_data()
    print(df.head())
    print("Dataset size:", len(df))