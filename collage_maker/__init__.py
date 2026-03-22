"""collage_maker — publication-quality image collage generators.

Modules
-------
make_collage
    7-tile asymmetric layout targeting ICCP single-column width.
make_collage_13imgs
    13-tile labelled layout with three vertical category strips.
"""

from collage_maker.make_collage import make_collage as make_collage_7
from collage_maker.make_collage_13imgs import make_collage as make_collage_13

__all__ = ["make_collage_7", "make_collage_13"]
