import json
import sys

def generate_latex_table(data):
    # LaTeX table header
    latex_table = """
\\begin{table}[h]
\\centering
\\begin{tabular}{|c|c|}
\\hline
LayerIndex & Score \\\\
\\hline
"""

    # Add rows to the table
    for item in data:
        layer_index, score = item['py/tuple']
        latex_table += f"{layer_index} & {score:.6f} \\\\\n\\hline\n"

    # LaTeX table footer
    latex_table += """\\end{tabular}
\\caption{Layer Scores}
\\end{table}
"""

    return latex_table

def main(input_file):
    # Read JSON data from file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Generate LaTeX table
    latex_table = generate_latex_table(data)

    # Write the LaTeX table to a .tex file
    output_file = "output_table.tex"
    with open(output_file, 'w') as file:
        file.write(latex_table)

    print(f"LaTeX table successfully generated and saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tablePlease.py file.json")
    else:
        input_file = sys.argv[1]
        main(input_file)
