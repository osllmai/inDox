import re
from datetime import datetime

# Read the _quarto.yml file
with open('_quarto.yml', 'r') as file:
    content = file.read()

# Update the last_modified field
new_content = re.sub(r'last_modified: ".*"', f'last_modified: "{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"', content)

# Write the updated content back to the _quarto.yml file
with open('_quarto.yml', 'w') as file:
    file.write(new_content)

# Render the Quarto project
import os
os.system('quarto render')
