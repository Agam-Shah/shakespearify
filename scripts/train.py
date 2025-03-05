from datasets import load_from_disk
import os


def load_processed_data():
    '''
    Load the processed data from the disk

    Returns:
        returns the processed data
    '''
    save_dir = os.path.join(os.getcwd(), "data", "processed")
    dataset = load_from_disk(save_dir)
    print(f"Loaded {len(dataset)} preprocessed samples")
    print(dataset[0])
    return dataset


if __name__ == '__main__':
    dataset = load_processed_data()