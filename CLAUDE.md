# Project rules
- Never commit files in data/ or models/, or any .env file.
- Never commit directly to main; work on a branch and open a pull request.
- The cutoff date, closure label, train/test split and evaluation function
  are defined once in src/ (settings in src/config.py) and imported everywhere.
- Add any new package to requirements.txt.
- Notebooks belong to one person each; put shared code in src/.
