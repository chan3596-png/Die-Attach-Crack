import pathlib

html = pathlib.Path('iii_die attatch/index.html').read_text(encoding='utf-8')
h = html.lower()

checks = {
    'MathJax':          'mathjax' in h,
    'Mermaid':          'mermaid' in h,
    'Base64_images':    'data:image/png;base64' in html,
    'Zoom_Modal':       'modal' in h,
    'Tooltip':          'tooltip' in h,
    'JS_Simulator':     'logit' in h or 'p(crack)' in h or 'simulator' in h,
    'polyfill_absent':  'polyfill.io' not in html,
    'Mermaid_diagram':  'graph lr' in h or 'graph td' in h or 'flowchart' in h or 'mermaid' in h,
    'References':       'reference' in h,
    'OR_formula':       'odds' in h or 'or=' in h,
}

print(f"File size: {len(html):,} bytes ({len(html)//1024} KB)\n")
all_ok = True
for k, v in checks.items():
    status = "✅ OK" if v else "❌ MISSING"
    if not v:
        all_ok = False
    print(f"  {status}  {k}")

print(f"\n{'All checks passed!' if all_ok else 'Some checks FAILED — review needed'}")
