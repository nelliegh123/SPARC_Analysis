#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CERN and the Peary Caribou authors
# SPDX-License-Identifier: LGPL-3.0-only


# IPHC data in the form (ireset_pcb, ireset_pixel)
IRESET_DATA = [
    (1e-06, 6.093e-12),
    (1.172e-06, 6.234e-12),
    (1.374e-06, 6.399e-12),
    (1.61e-06, 6.592e-12),
    (1.887e-06, 6.818e-12),
    (2.212e-06, 7.081e-12),
    (2.593e-06, 7.389e-12),
    (3.039e-06, 7.75e-12),
    (3.562e-06, 8.173e-12),
    (4.175e-06, 8.67e-12),
    (4.894e-06, 9.255e-12),
    (5.736e-06, 9.946e-12),
    (6.723e-06, 1.076e-11),
    (7.88e-06, 1.173e-11),
    (9.237e-06, 1.288e-11),
    (1.083e-05, 1.425e-11),
    (1.269e-05, 1.588e-11),
    (1.487e-05, 1.782e-11),
    (1.743e-05, 2.013e-11),
    (2.043e-05, 2.288e-11),
    (2.395e-05, 2.615e-11),
    (2.807e-05, 3.003e-11),
    (3.29e-05, 3.46e-11),
    (3.857e-05, 4e-11),
    (4.52e-05, 4.632e-11),
    (5.298e-05, 5.372e-11),
    (6.21e-05, 6.223e-11),
    (7.279e-05, 7.163e-11),
    (8.532e-05, 8.12e-11),
    (0.0001, 9.045e-11),
]


IRESET_PIXEL_MIN = 6e-12  # 6 pA
IRESET_PCB_MIN = 0e-6  # 0 uA
IRESET_PIXEL_MAX = 1.2e-10  # 120 pA
IRESET_PCB_MAX = 150e-6  # 150 uA


def _interpolate(
    table: list[tuple[float, float]],
    value: float,
    min_value: float,
    max_value: float,
) -> float:
    """Linearly interpolate ``value`` against a sorted (x, y) lookup table.

    ``value`` must lie within the physical range ``[min_value, max_value]``;
    otherwise a ``ValueError`` is raised, as interpolating outside the valid
    measurement range is not meaningful.
    """
    if value < min_value or value > max_value:
        raise ValueError(f"Input {value} is outside the valid range [{min_value}, {max_value}]")

    if value <= table[0][0]:
        x0, y0 = table[0]
        x1, y1 = table[1]
    elif value >= table[-1][0]:
        x0, y0 = table[-2]
        x1, y1 = table[-1]
    else:
        # Find the first entry whose x is greater than value; the previous
        # entry is the lower bound of the bracketing segment.
        idx = next(i for i in range(len(table)) if table[i][0] > value)
        x0, y0 = table[idx - 1]
        x1, y1 = table[idx]

    return y0 + (y1 - y0) * (value - x0) / (x1 - x0)


def ireset_pcb_to_ireset_pixel(ireset_pcb: float) -> float:
    """Convert PCB Ireset to pixel Ireset using the lookup table"""
    return _interpolate(IRESET_DATA, ireset_pcb, IRESET_PCB_MIN, IRESET_PCB_MAX)


def ireset_pixel_to_ireset_pcb(ireset_pixel: float) -> float:
    """Convert pixel Ireset to PCB Ireset using the inverted lookup table"""
    inverted_ireset_data = [(pixel, pcb) for pcb, pixel in IRESET_DATA]
    return _interpolate(inverted_ireset_data, ireset_pixel, IRESET_PIXEL_MIN, IRESET_PIXEL_MAX)


def vreset_to_ireset_pcb(vreset: float) -> float:
    """Convert Vreset to PCB Ireset"""
    return vreset / 20e3


def ireset_pcb_to_vreset(ireset_pcb: float) -> float:
    """Convert PCB Ireset to Vreset"""
    return ireset_pcb * 20e3


def vreset_to_ireset_pixel(vreset: float) -> float:
    """Convert Vreset to pixel Ireset"""
    return ireset_pcb_to_ireset_pixel(vreset_to_ireset_pcb(vreset))


def ireset_pixel_to_vreset(ireset_pixel: float) -> float:
    """Convert pixel Ireset to Vreset"""
    return ireset_pcb_to_vreset(ireset_pixel_to_ireset_pcb(ireset_pixel))


def main(cli_args: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser("sparc_ireset_calculator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    v2ipx = subparsers.add_parser("v2ipx", help="Calculate pixel Ireset for a given Vreset")
    v2ipx.add_argument("vreset", help="Vreset in V", type=float)
    v2ipcb = subparsers.add_parser("v2ipcb", help="Calculate PCB Ireset for a given Vreset")
    v2ipcb.add_argument("vreset", help="Vreset in V", type=float)
    ipcb2ipx = subparsers.add_parser("ipcb2ipx", help="Calculate pixel Ireset for a given PCB Ireset")
    ipcb2ipx.add_argument("ireset_pcb", help="PCB Ireset in uA", type=float)
    ipx2v = subparsers.add_parser("ipx2v", help="Calculate Vreset for a given pixel Ireset")
    ipx2v.add_argument("ireset_pixel", help="Pixel Ireset in pA", type=float)
    ipcb2v = subparsers.add_parser("ipcb2v", help="Calculate Vreset for a given PCB Ireset")
    ipcb2v.add_argument("ireset_pcb", help="PCB Ireset in uA", type=float)
    ipx2ipcb = subparsers.add_parser("ipx2ipcb", help="Calculate PCB Ireset for a given pixel Ireset")
    ipx2ipcb.add_argument("ireset_pixel", help="Pixel Ireset in pA", type=float)

    args = parser.parse_args(cli_args)

    if args.command == "v2ipx":
        print(f"Vreset = {args.vreset} V => Pixel Ireset = {vreset_to_ireset_pixel(args.vreset) * 1e12:.3g} pA")
    elif args.command == "v2ipcb":
        print(f"Vreset = {args.vreset} V => PCB Ireset = {vreset_to_ireset_pcb(args.vreset) * 1e6:.3g} uA")
    elif args.command == "ipcb2ipx":
        print(
            f"PCB Ireset = {args.ireset_pcb} uA => "
            f"Pixel Ireset = {ireset_pcb_to_ireset_pixel(args.ireset_pcb * 1e-6) * 1e12:.3g} pA"
        )
    elif args.command == "ipx2v":
        print(
            f"Pixel Ireset = {args.ireset_pixel} pA => "
            f"Vreset = {ireset_pixel_to_vreset(args.ireset_pixel * 1e-12):.3g} V"
        )
    elif args.command == "ipcb2v":
        print(f"PCB Ireset = {args.ireset_pcb} uA => Vreset = {ireset_pcb_to_vreset(args.ireset_pcb * 1e-6):.3g}")
    elif args.command == "ipx2ipcb":
        print(
            f"Pixel Ireset = {args.ireset_pixel} pA => "
            f"PCB Ireset = {ireset_pixel_to_ireset_pcb(args.ireset_pixel * 1e-12) * 1e6:.3g} uA"
        )


if __name__ == "__main__":
    main()
