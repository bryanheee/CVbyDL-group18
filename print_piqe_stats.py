import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def load_piqe_scores(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def calculate_statistics(scores):
    if not scores:
        return {
            'count': 0,
            'average': 0,
            'std_dev': 0,
            'variance': 0,
            'median': 0
        }
    
    stats = {
        'count': len(scores),
        'average': np.mean(scores),
        'std_dev': np.std(scores),
        'variance': np.var(scores),
        'median': np.median(scores)
    }
    return stats

def print_statistics(file_name, stats):
    print(f"\nStatistics for {file_name}:")
    print(f"Count: {stats['count']}")
    print(f"Average: {stats['average']:.2f} (Lower is better)")
    print(f"Standard Deviation: {stats['std_dev']:.2f}")
    print(f"Variance: {stats['variance']:.2f}")
    print(f"Median: {stats['median']:.2f}")

def save_boxplot(scores, file_name):
    plt.figure(figsize=(10, 6))
    plt.boxplot(scores, vert=False)
    plt.title(f"Boxplot for {file_name} (Lower PIQE is better)")
    plt.xlabel("PIQE Score")
    plt.savefig(f"boxplots/{file_name}_boxplot.png")
    plt.close()

def main():
    # Directory containing the PIQE score files
    piqe_dir = 'piqe_scores'
    
    # Ensure the boxplot directory exists
    os.makedirs("boxplots", exist_ok=True)
    
    # List of pickle files
    piqe_files = [
        "high_res_piqe_scores.pkl",
        "high_res_to_low_piqe_scores.pkl",
        "low_res_piqe_scores.pkl",
        "low_res_to_high_piqe_scores.pkl"
    ]
    
    for piqe_file in piqe_files:
        file_path = os.path.join(piqe_dir, piqe_file)
        if os.path.exists(file_path):
            scores = load_piqe_scores(file_path)
            stats = calculate_statistics(scores)
            print_statistics(piqe_file, stats)
            save_boxplot(scores, piqe_file.split(".")[0])  # Save boxplot
        else:
            print(f"File {file_path} does not exist.")

if __name__ == "__main__":
    main()