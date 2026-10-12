#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CERN and the Peary Caribou authors
# SPDX-License-Identifier: LGPL-3.0-only


NCOLS = 28
NROWS = 32
NPIXELS = NCOLS * NROWS


def sparc_mask_all() -> dict[str, int]:
    """Mask all pixels

    Returns:
        Dictionary containing register values
    """
    registers: dict[str, int] = {}
    for col in range((NCOLS + 1) // 2 + 1):
        registers[f"enable_col_{col}"] = 0
    for row in range(NROWS):
        registers[f"enable_row_{row}_even"] = 0
        registers[f"enable_row_{row}_odd"] = 0
    return registers


def sparc_mask_none() -> dict[str, int]:
    """Mask no pixels

    Returns:
        Dictionary containing register values
    """
    registers: dict[str, int] = {}
    for col in range((NCOLS + 1) // 2 + 1):
        registers[f"enable_col_{col}"] = 1
    for row in range(NROWS):
        registers[f"enable_row_{row}_even"] = 1
        registers[f"enable_row_{row}_odd"] = 1
    return registers


def sparc_mask_all_pixels_except(column: int, row: int) -> dict[str, int]:
    """Mask all pixels except the given one

    Arguments:
        column: pixel column
        row: pixel row

    Returns:
        Dictionary containing register values
    """
    registers = sparc_mask_all()
    enable_column = (column + 1) // 2
    registers[f"enable_col_{enable_column}"] = 1
    column_even_odd = "even" if column % 2 == 0 else "odd"
    registers[f"enable_row_{row}_{column_even_odd}"] = 1
    return registers


def sparc_mask_all_columns_except(column: int) -> dict[str, int]:
    """Mask all columns except the given one

    Arguments:
        column: single column (i.e. from 0 to 27)

    Returns:
        Dictionary containing register values
    """
    registers = sparc_mask_all()
    enable_column = (column + 1) // 2
    registers[f"enable_col_{enable_column}"] = 1
    column_even_odd = "even" if column % 2 == 0 else "odd"
    for row in range(NROWS):
        registers[f"enable_row_{row}_{column_even_odd}"] = 1
    return registers


def sparc_mask_pixels(pixels: list[tuple[int, int]]) -> dict[str, int]:
    """Mask all given pixels

    Uses a greedy low-collateral set cover to choose which registers to disable,
    repeatedly picking the register that masks the most remaining target pixels
    per total pixel masked, until every target is masked.

    Arguments:
        pixels: list of pixels given as (col, row)

    Returns:
        Dictionary containing register values
    """
    targets = set(pixels)

    # Build the candidate registers
    # Enumerate every register as a (register_name, footprint) candidate,
    # where footprint is the set of (col, row) pixels that disabling it will disable
    candidates: list[tuple[str, set[tuple[int, int]]]] = []

    # Row registers: one "even" and one "odd" register per row, each masking the
    # 14 pixels in that row whose column parity matches the register
    for row in range(NROWS):
        for even in (True, False):
            footprint = {(col, row) for col in range(0 if even else 1, NCOLS, 2)}
            candidates.append((f"enable_row_{row}_{'even' if even else 'odd'}", footprint))

    # Column registers: one register per single/double column, masking all rows of that column's pixel column(s)
    double_columns = ((0,),) + tuple((2 * n - 1, 2 * n) for n in range(1, NCOLS // 2)) + ((NCOLS - 1,),)
    for cols in double_columns:
        footprint = {(col, row) for col in cols for row in range(NROWS)}
        candidates.append((f"enable_col_{(cols[0] + 1) // 2}", footprint))

    # Greedy low-collateral set cover
    # Repeatedly choose the register that is "most worthwhile": the one masking the most not-yet-masked target pixels
    # per pixel it disables (best ratio) => this keeps collateral (pixels disabled beyond the targets) small
    # ``masked`` tracks pixels already disabled
    # ``covered`` counts how many distinct targets are disabled so far
    # ``chosen`` collects the register names to disable
    masked: set[tuple[int, int]] = set()
    covered = 0
    chosen: list[str] = []

    # Keep going until every requested pixel has been disabled
    while covered < len(targets):
        # Find the best candidate among all registers for this iteration
        best: str | None = None
        best_ratio = 0.0
        for name, footprint in candidates:
            # Only pixels not yet disabled count; already-masked footprint is useless
            new_pixels = footprint - masked
            if not new_pixels:
                continue
            # Of those, how many are target pixels still needing to be masked?
            new_targets = new_pixels & targets
            if not new_targets:
                continue
            # Ratio of useful pixels to total pixels disabled; higher = less collateral
            ratio = len(new_targets) / len(new_pixels)
            if ratio > best_ratio:
                best_ratio = ratio
                best = name

        # Every unmasked target is in its own row and column register footprint,
        # so ``best`` can never be None while targets remain unmasked
        assert best is not None, "greedy set cover failed to progress"

        # Commit to the winning register: remember it and mark its whole footprint
        # as disabled, counting any target pixels that just became disabled
        chosen.append(best)
        for name, footprint in candidates:
            if name == best:
                # Newly covered targets = targets in the footprint that were not yet masked
                covered += len(footprint & targets - masked)
                masked |= footprint
                break

    # Build the register dictionary
    # Start from "nothing masked" (every register enabled) and switch off exactly
    # the registers the greedy selected, so the result is a valid SPARC mask
    registers = sparc_mask_none()
    for name in chosen:
        registers[name] = 0
    return registers


def sparc_count_active_pixels(registers: dict[str, int]) -> int:
    """Count active pixel for a given set of registers

    Arguments:
        register: dictionary containing register values

    Returns:
        Number of active pixels
    """
    active_pixels = 0
    for col in range(NCOLS):
        col_register = f"enable_col_{(col + 1) // 2}"
        col_even_odd = "even" if col % 2 == 0 else "odd"
        for row in range(NROWS):
            row_register = f"enable_row_{row}_{col_even_odd}"
            if registers[row_register] == 1 and registers[col_register] == 1:
                active_pixels += 1
    return active_pixels


def main(cli_args: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser("sparc_masking_generator")
    parser.add_argument("-c", "--config", action="store_true", help="print output for config files")
    parser.add_argument("-z", "--only-zero", action="store_true", help="only report disabled registers")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("all", help="mask all pixels")
    subparsers.add_parser("none", help="mask no pixels")

    def _sparc_parse_pixel(value: str) -> tuple[int, int]:
        try:
            col, row = (int(part) for part in value.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value!r} is not a valid COL,ROW pair")
        if not (0 <= col < NCOLS and 0 <= row < NROWS):
            raise argparse.ArgumentTypeError(f"{value!r} is out of range: col 0-%d, row 0-%d" % (NCOLS - 1, NROWS - 1))
        return (col, row)

    pixels_except = subparsers.add_parser("pixels_except", help="mask all pixels except the given one")
    pixels_except.add_argument("pixel", type=_sparc_parse_pixel, metavar="COL,ROW", help="pixel to mask as COL,ROW")

    col_except = subparsers.add_parser("cols_except", help="mask all columns except the given one")
    col_except.add_argument("column", type=int, choices=range(NCOLS))

    pixels = subparsers.add_parser("pixels", help="mask the given pixels")
    pixels.add_argument(
        "pixels",
        type=_sparc_parse_pixel,
        nargs="+",
        metavar="COL,ROW",
        help="pixel to mask as COL,ROW (repeatable)",
    )

    args = parser.parse_args(cli_args)

    registers: dict[str, int] = {}
    if args.command == "all":
        registers = sparc_mask_all()
    elif args.command == "none":
        registers = sparc_mask_none()
    elif args.command == "pixels_except":
        registers = sparc_mask_all_pixels_except(*args.pixel)
    elif args.command == "cols_except":
        registers = sparc_mask_all_columns_except(args.column)
    elif args.command == "pixels":
        registers = sparc_mask_pixels(args.pixels)

    if args.config:
        for name, value in registers.items():
            if value == 1 and args.only_zero:
                continue
            print(f"{name} = {value}")
    else:
        print("Register values:")
        for name, value in registers.items():
            if value == 1 and args.only_zero:
                continue
            print(f"  {name:<18} = {value}")
        print()
        active_pixels = sparc_count_active_pixels(registers)
        masked_pixels = NPIXELS - active_pixels
        print(f"Masked pixels: {masked_pixels:>3} out of {NPIXELS} ({masked_pixels / NPIXELS * 100 :>4.1f}%)")
        print(f"Active pixels: {active_pixels:>3} out of {NPIXELS} ({active_pixels / NPIXELS * 100 :>4.1f}%)")


if __name__ == "__main__":
    main()
