import json

def generate_latex_table(json_filename):
    # Step 1: Read the JSON file
    with open(json_filename, 'r') as file:
        data = json.load(file)
    
    # Step 2: Initialize LaTeX table structure
    latex_table = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scalebox{0.5}{"
        "\\begin{tabular}{|c|c|c|c|c|c|}",
        "\\hline",
        "Token & Score & Token & Score & Token & Score \\\\",
        "\\hline"
    ]
    
    # Step 3: Read tokens and scores in chunks of 3
    tokens = list(data.keys())
    scores = list(data.values())
    
    for i in range(0, len(tokens), 3):
        row = []
        for j in range(3):
            if i + j < len(tokens):
                token = tokens[i + j]
                score = scores[i + j]
                row.append(f"{token} & {score:.6f}")
            else:
                # Fill with empty cells if tokens are less than 3
                row.append("&")
                row.append("&")
        
        # Join the row into LaTeX format
        latex_table.append(" & ".join(row) + " \\\\")
        latex_table.append("\\hline")
    
    # Step 4: End LaTeX table
    latex_table.append("\\end{tabular}")
    latex_table.append("}")
    latex_table.append("\\caption{Token Scores}")
    latex_table.append("\\end{table}")
    
    # Combine all parts of the LaTeX table into a single string
    latex_code = "\n".join(latex_table)
    
    # Print the LaTeX code
    print(latex_code)

# Call the function with the JSON file name
generate_latex_table('tokens.dump.json')