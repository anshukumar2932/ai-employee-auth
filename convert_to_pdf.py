#!/usr/bin/env python3
"""
Convert markdown to PDF using markdown2 and pdfkit or weasyprint
"""
import sys
import os

def convert_with_weasyprint():
    """Convert using weasyprint (recommended)"""
    try:
        from markdown import markdown
        from weasyprint import HTML, CSS
        
        # Read markdown
        with open('docs/AI-Powered-Contactless-Employee-Security-System-UPDATED.md', 'r') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown(md_content, extensions=['tables', 'fenced_code', 'toc'])
        
        # Add CSS styling
        html_with_style = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; margin-top: 20px; }}
                code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
                pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                @page {{ margin: 2cm; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        HTML(string=html_with_style).write_pdf('docs/AI-Powered-Contactless-Employee-Security-System.pdf')
        print("✓ PDF created successfully using weasyprint!")
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"Error with weasyprint: {e}")
        return False

def convert_with_pdfkit():
    """Convert using pdfkit"""
    try:
        import markdown
        import pdfkit
        
        # Read markdown
        with open('docs/AI-Powered-Contactless-Employee-Security-System-UPDATED.md', 'r') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Convert to PDF
        pdfkit.from_string(html_content, 'docs/AI-Powered-Contactless-Employee-Security-System.pdf')
        print("✓ PDF created successfully using pdfkit!")
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"Error with pdfkit: {e}")
        return False

if __name__ == '__main__':
    print("Attempting to convert markdown to PDF...")
    print()
    
    # Try weasyprint first (better quality)
    if convert_with_weasyprint():
        sys.exit(0)
    
    # Try pdfkit
    if convert_with_pdfkit():
        sys.exit(0)
    
    # No converter available
    print("ERROR: No PDF converter library found!")
    print()
    print("Please install one of the following:")
    print("1. weasyprint (recommended): pip install markdown weasyprint")
    print("2. pdfkit: pip install markdown pdfkit")
    print()
    print("Or use the bash script with pandoc:")
    print("   sudo apt install pandoc texlive-xetex")
    print("   ./convert_to_pdf.sh")
    print()
    print("Or use an online converter:")
    print("   - https://www.markdowntopdf.com/")
    print("   - https://cloudconvert.com/md-to-pdf")
    print()
    print("Input file: docs/AI-Powered-Contactless-Employee-Security-System-UPDATED.md")
    sys.exit(1)
