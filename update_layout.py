import sys
import textwrap

with open('src/modules.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Change the columns gap and add CSS
old_cols = 'left_col, right_col = st.columns([2.2, 1], gap="medium")'
new_cols = '''    st.markdown("""
        <style>
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.01) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 24px !important;
            padding: 24px !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    left_col, right_col = st.columns([2.2, 1], gap="small")'''
content = content.replace(old_cols, new_cols)

# 2. Extract Stat Row
stat_row_start = content.find('    # ── Stat Row ──')
stat_row_end = content.find('    # ── Main area + right panel ──')
stat_row_code = content[stat_row_start:stat_row_end]

# Remove stat row from original position
content = content[:stat_row_start] + content[stat_row_end:]

# 3. Handle with left_col:
left_col_start = content.find('    with left_col:')
right_col_start = content.find('    with right_col:')

if left_col_start == -1 or right_col_start == -1:
    print('Failed to find columns')
    sys.exit(1)

left_col_code = content[left_col_start:right_col_start]
# Indent the contents of left_col_code (skipping the first line '    with left_col:')
left_col_lines = left_col_code.split('\n')
indented_left_col = left_col_lines[0] + '\n        with st.container(border=True):\n'
# Also inject the stat row here!
indented_stat_row = textwrap.indent(stat_row_code, '    ')
indented_left_col += indented_stat_row
for line in left_col_lines[1:]:
    if line.strip() == '':
        indented_left_col += '\n'
    else:
        indented_left_col += '    ' + line + '\n'

# 4. Handle with right_col:
# It goes from right_col_start to the end of the function. The function ends before MODULE 2.
end_of_func = content.find('# ==============================================================================', right_col_start)
right_col_code = content[right_col_start:end_of_func]

right_col_lines = right_col_code.split('\n')
indented_right_col = right_col_lines[0] + '\n        with st.container(border=True):\n'
for line in right_col_lines[1:]:
    if line.strip() == '':
        indented_right_col += '\n'
    else:
        indented_right_col += '    ' + line + '\n'

# Replace in content
final_content = content[:left_col_start] + indented_left_col + indented_right_col + content[end_of_func:]

with open('src/modules.py', 'w', encoding='utf-8') as f:
    f.write(final_content)

print('Success')
