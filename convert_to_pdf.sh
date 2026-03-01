#!/bin/bash
# Script to convert the updated markdown document to PDF

echo "Converting markdown to PDF..."

# Check if pandoc is installed
if command -v pandoc &> /dev/null; then
    echo "Using Pandoc..."
    pandoc docs/AI-Powered-Contactless-Employee-Security-System-UPDATED.md \
        -o docs/AI-Powered-Contactless-Employee-Security-System.pdf \
        --pdf-engine=xelatex \
        --toc \
        --toc-depth=3 \
        --number-sections \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        -V documentclass=report \
        -V colorlinks=true \
        -V linkcolor=blue \
        -V urlcolor=blue
    echo "PDF created successfully!"
    
elif command -v markdown-pdf &> /dev/null; then
    echo "Using markdown-pdf..."
    markdown-pdf docs/AI-Powered-Contactless-Employee-Security-System-UPDATED.md \
        -o docs/AI-Powered-Contactless-Employee-Security-System.pdf
    echo "PDF created successfully!"
    
else
    echo "ERROR: No PDF converter found!"
    echo ""
    echo "Please install one of the following:"
    echo "1. Pandoc (recommended): sudo apt install pandoc texlive-xetex"
    echo "2. markdown-pdf: npm install -g markdown-pdf"
    echo ""
    echo "Or use online converter:"
    echo "- https://www.markdowntopdf.com/"
    echo "- https://cloudconvert.com/md-to-pdf"
    echo ""
    echo "Input file: docs/AI-Powered-Contactless-Employee-Security-System-UPDATED.md"
    exit 1
fi
