# Print the generated VHDL files in compile order: every entity before the files that use it.
#
# Vivado's xvhdl compiles files in the order it gets them, and a file that instantiates
# "entity work.x" fails if x is not compiled yet. The design is nested (transformer ->
# encoder -> encoderlayer_0 -> mha -> softmax -> softmax_ram ...), and the folder layout
# does not give that order, so it is read from the files themselves.
#
# Run from software/:   uv run python -m hw_design.vhdl_order ../hardware/qat6_seed1/source > files.txt

import re
import sys
from pathlib import Path

DEFINES = re.compile(r"^\s*entity\s+(\w+)\s+is", re.M)   # entity input_linear is
USES = re.compile(r"entity\s+\w+\.(\w+)")                 # entity work.input_linear(rtl)


def compile_order(folder):
    files = sorted(Path(folder).rglob("*.vhd"))
    defined_in, uses = {}, {}
    for f in files:
        text = f.read_text().lower()                      # VHDL is not case-sensitive
        for entity in DEFINES.findall(text):
            defined_in[entity] = f
        uses[f] = set(USES.findall(text))

    order, done = [], set()

    def visit(f):
        # depth-first: first everything f uses, then f itself (a topological sort)
        if f in done:
            return
        done.add(f)
        for entity in sorted(uses[f]):
            if entity in defined_in:
                visit(defined_in[entity])
        order.append(f)

    for f in files:
        visit(f)
    return order


if __name__ == "__main__":
    for f in compile_order(sys.argv[1]):
        print(f.resolve())        # absolute paths: the list works from any folder
