# AI-tampered-image-checker
A Python-based tool that analyzes images to detect whether they are AI-generated or real using metadata inspection, statistical analysis, frequency-domain forensics (DCT/DWT), and heuristic scoring.
It provides:

AI vs Real verdict with a confidence score (0–100%)

Full EXIF metadata extraction

Checks for AI artifacts, smoothness, symmetry, aspect-ratio patterns

DCT-based periodicity detection for AI diffusion fingerprints

Wavelet analysis (DWT) for hidden watermark or generative patterns

LSB steganography detection

Optional export to JSON report

Simple CLI interface using input() for easy use

Ideal for researchers, security analysts, or anyone wanting to verify an image’s authenticity.
