import json
import os
from datetime import datetime

def update_leaderboard():
    # Loading the scores from the root scores.json
    try:
        with open('scores.json', 'r') as f:
            scores = json.load(f)
    except FileNotFoundError:
        print("scores.json not found!")
        return

    # Sorting scores by accuracy (descending)
    scores.sort(key=lambda x: float(x['accuracy'].strip('%')), reverse=True)

    # Creating the Markdown Table string
    table_header = "| Rank | Participant | Accuracy | F1 (macro) | Date |\n"
    table_header += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    table_rows = ""
    for i, entry in enumerate(scores):
        # Add trophy emojis for top 3
        rank = i + 1
        if rank == 1: rank_str = "1"
        elif rank == 2: rank_str = "2"
        elif rank == 3: rank_str = "3"
        else: rank_str = str(rank)
        
        table_rows += f"| {rank_str} | {entry['participant']} | {entry['accuracy']} | {entry['f1_macro']} | {entry['date']} |\n"

    # Preparing the full README content
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    content = f"""# Flowers competition Leaderboard

*Last updated: {now}*

**Total submissions: {len(scores)}** | **Participants: {len(set(s['participant'] for s in scores))}**

{table_header}{table_rows}
"""

    #  Writing to the leaderboard README
    with open('leaderboard/README.md', 'w') as f:
        f.write(content)
    
    print("Leaderboard updated successfully!")

if __name__ == "__main__":
    update_leaderboard()
