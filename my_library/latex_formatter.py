import numpy as np

def generate_table(lists_of_data, labels, caption=None, output_path=None):
    """
    Generate a LaTeX table from a dictionary of data.

    Args:
        lists_of_data (list): A list of lists, where each inner list represents a column of data.
        labels (list): A list of strings representing the column names.
        caption (str, optional): The caption for the table.
        output_path (str, optional): The path to save the generated LaTeX code.

    Returns:
        str: A string containing the LaTeX code for the table.
    """
    # Extract column names and rows
    columns = list(labels)
    print(f"Columns: {columns}")
    print(f"There are [{len(columns)}] columns and [{len(lists_of_data)}] lists of data.")
    
    print(f"Rows: {len(list(lists_of_data)[0])}.")

    # Start building the LaTeX table
    latex_table = "\\begin{tabular}{" + " | ".join(["l"] * len(columns)) + "}\n"
    latex_table += " \\hline\n"
    
    # Add header row
    latex_table += " & ".join(columns) + " \\\\\n"
    latex_table += " \\hline\n"

    # Add data rows
    rows = zip(*lists_of_data)
    for row in rows:
        latex_table += " & ".join(map(str, row)) + " \\\\\n"
        latex_table += " \\hline\n"
        print(f"Row: {row}")  # Debugging line to check the content of each row

    # End the table
    latex_table += "\\end{tabular}\n"

    if caption:
        latex_table = f"\\begin{{table}}[h]\n\\centering\n{latex_table}\\caption{{{caption}}}\n\\end{{table}}\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_table)
    return latex_table