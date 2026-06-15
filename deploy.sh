#!/bin/bash

# Clear terminal screen
clear
echo "=================================================="
echo "🚀 Starting Git Deployment for Shyam M"
echo "=================================================="

# 1. Initialize Git repository if not already done
if [ ! -d ".git" ]; then
    echo "--> Initializing a brand new Git repository..."
    git init
fi

# 2. Add all project updates and structural files
echo "--> Staging repository updates..."
git add .

# 3. Prompt user for a custom commit message
echo -n "📝 Enter your commit message (or press Enter for default): "
read commit_msg

if [ -z "$commit_msg" ]; then
    commit_msg="Update: Refined project code and repository structure"
fi

# 4. Commit changes
git commit -m "$commit_msg"

# 5. Ensure the main branch is set correctly
git branch -M main

# 6. Check if the remote repository link is already set up
if ! git remote | grep -q "origin"; then
    echo "--> Linking your local project to GitHub..."
    git remote add origin https://github.com/shyam-m/CodeAlpha_Titanic_Classification.git
fi

# 7. Push live to your GitHub URL
echo "--> Pushing your codebase live to GitHub..."
git push -u origin main

echo "=================================================="
echo "✅ Deployment Finished! Your repository is updated."
echo "=================================================="
