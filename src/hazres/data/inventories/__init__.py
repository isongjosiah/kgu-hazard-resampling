"""Hazard inventory loaders. Importing this package registers them.

``ge_lucas``      GE-LUCAS v1.1 survey points (gullies, EU)
``vector``        any point or polygon file (landslides, African gully heads, ...)
``flood_raster``  Global Flood Database events exported as GeoTIFFs
"""

from hazres.data.inventories import flood_raster, ge_lucas, vector

__all__ = ["flood_raster", "ge_lucas", "vector"]
