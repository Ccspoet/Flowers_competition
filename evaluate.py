import os
import io
import sys
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def run_gradinimport os
import io
import sys
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def update_readme():
    """Reads scores.json and updates the README.md table automatically."""
    if not os.path.exists('scores.json'):
        return
        
    with open('scores.json', 'r') as f:
        scores = json.load(f)
    
    # Create the Markdown table header
    table = "| Rank | User | F1-Macro | Accuracy |\n| :--- | :--- | :--- | :--- |\n"
    
    for i, s in enumerate(scores, 1):
        table += f"| {i} | {s['user']} | {s['f1_macro']} | {s['accuracy']} |\n"
    
    with open('README.md', 'r') as f:
        readme = f.read()

    # Look for the marker to swap the table
    marker = "## 🏆 Leaderboard"
    if marker in readme:
        parts = readme.split(marker)
        # Keep everything before the marker, then append the new table
        # We split and take [0] to ensure we don't duplicate tables if the script runs twice
        new_readme = parts[0] + marker + "\n\n" + table
        with open('README.md', 'w') as f:
            f.write(new_readme)
        print("README.md leaderboard updated!")

def update_leaderboard(username, f1, acc):
    """Updates the JSON database and sorts by performance."""
    scores_file = 'scores.json'
    if os.path.exists(scores_file):
        with open(scores_file, 'r') as f:
            scores = json.load(f)
    else:
        scores = []

    # Remove old score for this user if it exists
    scores = [s for s in scores if s['user'] != username]
    
    # Append new score
    scores.append({
        "user": username, 
        "f1_macro": round(f1, 4), 
        "accuracy": round(acc, 4)
    })
    
    # Sort by F1 descending (Highest first)
    scores = sorted(scores, key=lambda x: x['f1_macro'], reverse=True)

    with open(scores_file, 'w') as f:
        json.dump(scores, f, indent=4)

def run_grading(student_username):
    # 1. Load the Secret Master List
    secret_data = os.getenv("FLOWER_ANSWERS")
    if not secret_data:
        print("Error: Secret answer key (FLOWER_ANSWERS) missing from GitHub Secrets.")
        sys.exit(1)
    
    truth_df = pd.read_csv(io.StringIO(secret_data), names=['original_filename', 'label'])
    truth_df['id'] = range(len(truth_df))

    # 2. Load Student Submission
    try:
        sub_df = pd.read_csv('submission.csv')
    except Exception as e:
        print(f"Error reading submission.csv: {e}")
        sys.exit(1)

    # 3. Smart Matching Logic
    if 'original_filename' in sub_df.columns:
        merged = pd.merge(truth_df, sub_df, on='original_filename', suffixes=('_true', '_pred'))
        y_true = merged['label_true']
        y_pred = merged['label_pred']
    elif 'id' in sub_df.columns:
        merged = pd.merge(truth_df, sub_df, on='id', suffixes=('_true', '_pred'))
        y_true = merged['label_true']
        y_pred = merged['label_pred']
    elif 'label' in sub_df.columns:
        if len(sub_df) != len(truth_df):
            print("Error: Label-only submission length mismatch.")
            sys.exit(1)
        y_true = truth_df['label']
        y_pred = sub_df['label']
    else:
        print("Error: Submission must have 'original_filename', 'id', or 'label' columns.")
        sys.exit(1)

    # 4. Calculate Metrics
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    # 5. Update Database and README
    update_leaderboard(student_username, f1, acc)
    update_readme()

if __name__ == "__main__":
    user_arg = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    run_grading(user_arg)g(student_username):
    # 1. Load the Secret Master List (Always filename + label)
    secret_data = os.getenv("FLOWER_ANSWERS")
    if not secret_data:
        print("Error: Secret answer key missing.")
        sys.exit(1)
    
    # We load your secret and create a helper 'id' column based on row order
    truth_df = pd.read_csv(io.StringIO(secret_data), names=['original_filename', 'label'])
    truth_df['id'] = range(len(truth_df))

    # 2. Load Student Submission
    try:
        sub_df = pd.read_csv('submission.csv')
    except Exception as e:
        print(f"Error reading submission.csv: {e}")
        sys.exit(1)

    # 3. Smart Matching Logic
    # Scenario A: Filename and Label provided
    if 'original_filename' in sub_df.columns:
        merged = pd.merge(truth_df, sub_df, on='original_filename', suffixes=('_true', '_pred'))
        y_true = merged['label_true']
        y_pred = merged['label_pred']

    # Scenario B: ID and Label provided
    elif 'id' in sub_df.columns:
        merged = pd.merge(truth_df, sub_df, on='id', suffixes=('_true', '_pred'))
        y_true = merged['label_true']
        y_pred = merged['label_pred']

    # Scenario C: Only 'label' provided
    elif 'label' in sub_df.columns:
        if len(sub_df) != len(truth_df):
            print("Error: Label-only submission length does not match test set.")
            sys.exit(1)
        y_true = truth_df['label']
        y_pred = sub_df['label']
    
    else:
        print("Error: Submission must have 'original_filename', 'id', or at least 'label' column.")
        sys.exit(1)

    # 4. Calculate Metrics
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    # 5. Update Database
    update_leaderboard(student_username, f1, acc)

def update_leaderboard(username, f1, acc):
    scores_file = 'scores.json'
    if os.path.exists(scores_file):
        with open(scores_file, 'r') as f:
            scores = json.load(f)
    else:
        scores = []

    # Update logic
    scores = [s for s in scores if s['user'] != username]
    scores.append({
        "user": username, 
        "f1_macro": round(f1, 4), 
        "accuracy": round(acc, 4)
    })
    
    # Sort by F1 descending
    scores = sorted(scores, key=lambda x: x['f1_macro'], reverse=True)

    with open(scores_file, 'w') as f:
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    user = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    run_grading(user)

def update_readme():
    with open('scores.json', 'r') as f:
        scores = json.load(f)
    
    # Create the Markdown table header
    table = "| Rank | User | F1-Macro | Accuracy |\n| :--- | :--- | :--- | :--- |\n"
    
    for i, s in enumerate(scores, 1):
        table += f"| {i} | {s['user']} | {s['f1_macro']} | {s['accuracy']} |\n"
    
    with open('README.md', 'r') as f:
        readme = f.read()

    # This looks for a special marker in your README to swap the table
    marker = "## 🏆 Leaderboard"
    if marker in readme:
        parts = readme.split(marker)
        # Keep everything before the marker, then add the marker and the new table
        new_readme = parts[0] + marker + "\n\n" + table
        with open('README.md', 'w') as f:
            f.write(new_readme)

# Call this at the very end of your run_grading function
