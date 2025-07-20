#!/bin/bash
# https://chatgpt.com/share/687b497d-908c-8008-b0fd-3b67d22800b2
set -e  # Stop script on error

# ====== USER CONFIGURATION ======
REPO_NAME="AAIPL_CDS_CSA"      # <-- Change this to actual IP
VISIBILITY="public"                    # or "private"
COMMIT_MESSAGE="Initial commit"
GITHUB_USERNAME="shivaiisc"        # <-- Change this
GITHUB_EMAIL="shivac@iisc.ac.in"         # <-- Change this
# =================================

git config --global credential.helper store
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"

git add .
git commit -m "$COMMIT_MESSAGE"
git branch -M main

# Step 5: Set remote & push
REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
git remote add origin "$REMOTE_URL"

echo "Pushing code to GitHub..."
git push -u origin main

echo "✅ Done! Repo pushed to: $REMOTE_URL"
