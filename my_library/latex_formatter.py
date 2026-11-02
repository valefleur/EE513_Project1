import numpy as np

_LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                  "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}


def latex_escape(value):
    """Escape the characters LaTeX treats as commands, e.g. the _ in APC_0223_white.dng
    (an unescaped _ stops compilation with 'Missing $ inserted')."""
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(value))


def generate_table(lists_of_data, labels, caption=None, output_path=None,
                   escape=False, longtable=False, VERBOSE=True):
    """
    Generate a LaTeX table from a dictionary of data.

    Args:
        lists_of_data (list): A list of lists, where each inner list represents a column of data.
        labels (list): A list of strings representing the column names.
        caption (str, optional): The caption for the table.
        output_path (str, optional): The path to save the generated LaTeX code.
        escape (bool): escape LaTeX special characters in labels and cells.
        longtable (bool): emit a longtable (needs \\usepackage{longtable}) so a table
            longer than a page breaks across pages; the caption goes inside it.
        VERBOSE (bool): print the columns and every row while building.

    Returns:
        str: A string containing the LaTeX code for the table.
    """
    # Extract column names and rows
    columns = list(labels)
    if VERBOSE:
        print(f"Columns: {columns}")
        print(f"There are [{len(columns)}] columns and [{len(lists_of_data)}] lists of data.")
        print(f"Rows: {len(list(lists_of_data)[0])}.")
    fmt = latex_escape if escape else str

    # Start building the LaTeX table
    env = "longtable" if longtable else "tabular"
    latex_table = f"\\begin{{{env}}}{{" + " | ".join(["l"] * len(columns)) + "}\n"
    if longtable and caption:
        latex_table += f"\\caption{{{caption}}} \\\\\n"
    latex_table += " \\hline\n"

    # Add header row
    latex_table += " & ".join(map(fmt, columns)) + " \\\\\n"
    latex_table += " \\hline\n"
    if longtable:
        latex_table += "\\endhead\n"          # repeat the header on every page

    # Add data rows
    rows = zip(*lists_of_data)
    for row in rows:
        latex_table += " & ".join(map(fmt, row)) + " \\\\\n"
        latex_table += " \\hline\n"
        if VERBOSE:
            print(f"Row: {row}")  # Debugging line to check the content of each row

    # End the table
    latex_table += f"\\end{{{env}}}\n"

    if caption and not longtable:
        latex_table = f"\\begin{{table}}[h]\n\\centering\n{latex_table}\\caption{{{caption}}}\n\\end{{table}}\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_table)
    return latex_table