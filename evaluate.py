name: Update Leaderboard
on:
  pull_request_target:
    types: [opened, synchronize]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    defaults:
      run:
        working-directory: ${{ github.workspace }}
    steps:
      - name: Checkout Main
        uses: actions/checkout@v3
        with:
          ref: main

      - name: Checkout Student PR
        uses: actions/checkout@v3
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          path: student_submission

      - name: Load Student Data
        run: |
          if [ -f "student_submission/submission.csv" ]; then
            mkdir -p submissions
            cp student_submission/submission.csv ./submissions/submission.csv
            echo "Submission found and loaded."
            ls -la submissions/
          else
            echo "Error: submission.csv not found in the student's PR."
            find student_submission/ -type f
            exit 1
          fi

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install pandas scikit-learn

      - name: Run Grading
        env:
          FLOWER_ANSWERS: ${{ secrets.FLOWER_ANSWERS }}
        run: |
          echo "=== Working directory ==="
          pwd
          echo "=== Files ==="
          ls -la
          echo "=== Submissions folder ==="
          ls -la submissions/
          python evaluate.py ${{ github.event.pull_request.user.login }}

      - name: Push Results
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "Leaderboard Bot"
          git add scores.json README.md
          git commit -m "🏆 Score for ${{ github.event.pull_request.user.login }}" || echo "No changes"
          git push origin main
