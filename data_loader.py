import pandas as pd
import json

def load_and_create_mini_dataset(file_path, mini_file_path, max_records=10000):
    """
    Reads the large JSON file, creates a smaller mini-dataset file,
    and returns a DataFrame of the mini-dataset.
    """
    data = []
    print(f"Reading first {max_records} records from file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"Successfully loaded {len(data)} valid reviews.")

    # Save the mini dataset
    print(f"Saving to {mini_file_path}...")
    with open(mini_file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')

    print(f"Mini dataset saved to {mini_file_path}!")

    # Create DataFrame
    df = pd.DataFrame(data)
    print(f"DataFrame created! Shape: {df.shape}")
    return df

def load_mini_dataset(mini_file_path):
    """
    Loads the mini dataset from a pre-existing JSON file.
    """
    print(f"Loading mini dataset from {mini_file_path}...")
    df = pd.read_json(mini_file_path, lines=True)
    print(f"Mini dataset loaded! Shape: {df.shape}")
    return df
