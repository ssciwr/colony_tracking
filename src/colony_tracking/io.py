"""Input-tree discovery and shared I/O.

Discovers both accepted input tree layouts (flat and per-well), parses
well/frame/time metadata from file names, applies the shared 1/99 percentile
stretch, and reads and writes label images, CSVs, done markers, and scale
files (including the prototype's legacy column names). Implemented in a
later ticket.
"""
