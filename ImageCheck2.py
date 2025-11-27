"""
Image Forensics Analyzer (fixed)
- Prompts user for image path using input()
- Keeps optional scipy / pywt usage when available
- Computes a final AI-vs-Real verdict with a confidence score (0-100)
- Exports full results and final verdict to JSON if requested

Fixes applied:
- Corrected conditional import for scipy.signal.convolve2d
- Added safe fallbacks when scipy/pywt are not available
- Minor robustness improvements
"""

import os
import sys
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
from pathlib import Path
import hashlib
import json

# Optional imports for advanced analysis
try:
    from scipy.fftpack import dct, idct
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Conditionally import convolve2d if scipy is available
if SCIPY_AVAILABLE:
    try:
        from scipy.signal import convolve2d as _conv2d
    except Exception:
        _conv2d = None
else:
    _conv2d = None

try:
    import pywt
    PYWT_AVAILABLE = True
except Exception:
    PYWT_AVAILABLE = False


class ImageForensicsAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.results = {}
        # store intermediate flags used for final scoring
        self._flags = {}

    def analyze_all(self):
        """Run all analysis methods"""
        print(f"\n{'='*60}")
        print(f"Analyzing: {self.image_path}")
        print(f"{'='*60}\n")

        self.extract_basic_info()
        self.extract_metadata()
        self.analyze_statistical_properties()
        self.detect_ai_artifacts()
        self.check_for_watermarks()
        self.calculate_hashes()
        # compute final verdict
        self.compute_final_verdict()

        return self.results

    def extract_basic_info(self):
        """Extract basic image information"""
        print("📋 BASIC INFORMATION")
        print("-" * 60)

        info = {
            'filename': os.path.basename(self.image_path),
            'format': self.image.format,
            'mode': self.image.mode,
            'size': self.image.size,
            'width': self.image.width,
            'height': self.image.height,
            'megapixels': round((self.image.width * self.image.height) / 1_000_000, 2),
            'file_size_mb': round(os.path.getsize(self.image_path) / (1024 * 1024), 2)
        }

        self.results['basic_info'] = info

        for key, value in info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print()

    def extract_metadata(self):
        """Extract EXIF and other metadata"""
        print("🔍 METADATA (EXIF)")
        print("-" * 60)

        exif_data = {}

        try:
            exif = self.image._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)

                    # Handle GPS data specially
                    if tag == "GPSInfo":
                        gps_data = {}
                        for gps_tag_id in value:
                            gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            gps_data[gps_tag] = value[gps_tag_id]
                        exif_data[tag] = gps_data
                    else:
                        # Convert bytes to string for readability
                        if isinstance(value, bytes):
                            try:
                                value = value.decode()
                            except Exception:
                                value = str(value)
                        exif_data[tag] = value

                self.results['metadata'] = exif_data

                # Print important metadata
                important_tags = ['Make', 'Model', 'Software', 'DateTime',
                                  'DateTimeOriginal', 'Artist', 'Copyright',
                                  'UserComment', 'ImageDescription']

                for tag in important_tags:
                    if tag in exif_data:
                        print(f"  {tag}: {exif_data[tag]}")

                # Check for AI generation indicators in metadata
                ai_indicators = ['AI', 'Stable Diffusion', 'DALL-E', 'Midjourney',
                                 'Flux', 'Generated', 'Synthetic']

                print("\n  🤖 AI Generation Indicators in Metadata:")
                found_indicators = []
                for tag, value in exif_data.items():
                    value_str = str(value).lower()
                    for indicator in ai_indicators:
                        if indicator.lower() in value_str:
                            found_indicators.append(f"{tag}: {value}")

                if found_indicators:
                    for indicator in found_indicators:
                        print(f"    ⚠️  {indicator}")
                else:
                    print("    ✓ None found")
            else:
                print("  No EXIF data found")
                self.results['metadata'] = None

        except AttributeError:
            print("  No EXIF data available for this image format")
            self.results['metadata'] = None
        print()

    def analyze_statistical_properties(self):
        """Analyze statistical properties that might indicate AI generation"""
        print("📊 STATISTICAL ANALYSIS")
        print("-" * 60)

        img_array = np.array(self.image.convert('RGB'))

        stats = {
            'mean_rgb': np.mean(img_array, axis=(0, 1)).tolist(),
            'std_rgb': np.std(img_array, axis=(0, 1)).tolist(),
            'min_value': int(np.min(img_array)),
            'max_value': int(np.max(img_array)),
        }

        # Calculate histogram entropy (measure of randomness)
        hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        stats['entropy'] = round(entropy, 3)

        # Check for suspicious patterns (perfect pixels, unusual distributions)
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        stats['unique_colors'] = unique_colors
        stats['color_diversity_ratio'] = round(unique_colors / total_pixels, 4)

        self.results['statistics'] = stats

        print(f"  Entropy: {stats['entropy']} (higher = more random)")
        print(f"  Unique Colors: {stats['unique_colors']:,}")
        print(f"  Color Diversity: {stats['color_diversity_ratio']}")
        print(f"  Mean RGB: {[round(x, 1) for x in stats['mean_rgb']]}")
        print()

    def detect_ai_artifacts(self):
        """Look for common AI generation artifacts"""
        print("🎨 AI GENERATION ARTIFACTS")
        print("-" * 60)

        img_array = np.array(self.image.convert('RGB'))
        artifacts = []

        # 1. Check for unusual smoothness (AI images often lack high-frequency noise)
        gray = np.mean(img_array, axis=2)
        # Use a simple Laplacian kernel via convolution on 2D
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        try:
            if _conv2d is not None:
                # convolution expects 2D arrays
                edges = np.abs(_conv2d(gray, laplacian_kernel, mode='same'))
            else:
                # fallback simple finite-difference approximation
                gx = np.diff(gray, axis=1, append=gray[:, -1:])
                gy = np.diff(gray, axis=0, append=gray[-1:, :])
                edges = np.hypot(gx, gy)
        except Exception:
            # safest fallback
            gx = np.diff(gray, axis=1, append=gray[:, -1:])
            gy = np.diff(gray, axis=0, append=gray[-1:, :])
            edges = np.hypot(gx, gy)

        edge_variance = float(np.var(edges))

        if edge_variance < 100:
            artifacts.append("⚠️  Low edge variance (unnaturally smooth)")
            self._flags['low_edge_variance'] = True
        else:
            artifacts.append("✓ Normal edge variance")
            self._flags['low_edge_variance'] = False

        # 2. Check aspect ratio (AI often generates square or other common ratios)
        width, height = self.image.size
        ratio = width / height
        common_ai_ratios = [1.0, 1.33, 1.5, 1.78, 0.75, 0.67]

        if any(abs(ratio - r) < 0.05 for r in common_ai_ratios):
            artifacts.append(f"⚠️  Common AI aspect ratio: {ratio:.2f}")
            self._flags['common_ai_aspect_ratio'] = True
        else:
            self._flags['common_ai_aspect_ratio'] = False

        # 3. Check for PNG dimensions pattern (heuristic)
        if self.image.format == 'PNG' and self.image.width % 64 == 0:
            artifacts.append("⚠️  PNG format with 64-divisible dimensions (common in AI)")
            self._flags['png_64_divisible'] = True
        else:
            self._flags['png_64_divisible'] = False

        # 4. Check for symmetry (quick heuristic)
        left_half = img_array[:, :width//2]
        right_half = np.fliplr(img_array[:, width//2:width//2 + left_half.shape[1]])
        symmetry = None
        if left_half.shape == right_half.shape:
            try:
                symmetry = float(np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1])
            except Exception:
                symmetry = 0.0
            if symmetry > 0.95:
                artifacts.append(f"⚠️  High symmetry detected: {symmetry:.3f}")
                self._flags['high_symmetry'] = True
            else:
                self._flags['high_symmetry'] = False
        else:
            self._flags['high_symmetry'] = False

        self.results['ai_artifacts'] = artifacts

        for artifact in artifacts:
            print(f"  {artifact}")
        print()

    def check_for_watermarks(self):
        """Check for visible and statistical watermark patterns"""
        print("💧 WATERMARK DETECTION")
        print("-" * 60)

        watermark_indicators = []

        # 1. Check for uniform patterns in LSBs (Least Significant Bits)
        img_array = np.array(self.image.convert('RGB'))
        lsb = img_array & 1  # Extract least significant bit
        lsb_entropy = self._calculate_entropy(lsb.flatten())

        if lsb_entropy > 0.9:
            watermark_indicators.append("⚠️  High LSB entropy (possible steganography)")
            self._flags['high_lsb_entropy'] = True
        else:
            watermark_indicators.append("✓ Normal LSB patterns")
            self._flags['high_lsb_entropy'] = False

        # 2. DCT Analysis (Discrete Cosine Transform)
        if SCIPY_AVAILABLE and _conv2d is not None:
            dct_results = self._analyze_dct_watermarks(img_array)
            watermark_indicators.extend(dct_results)
            # store peak ratio for scoring
            self._flags['dct_peak_to_avg'] = self.results.get('dct_analysis', {}).get('peak_to_avg_ratio', 0)
        elif SCIPY_AVAILABLE and _conv2d is None:
            # SciPy present but convolve2d failed to import - still attempt DCT if fftpack available
            try:
                dct_results = self._analyze_dct_watermarks(img_array)
                watermark_indicators.extend(dct_results)
                self._flags['dct_peak_to_avg'] = self.results.get('dct_analysis', {}).get('peak_to_avg_ratio', 0)
            except Exception:
                watermark_indicators.append("ℹ️  SciPy installed but DCT analysis failed")
                self._flags['dct_peak_to_avg'] = 0
        else:
            watermark_indicators.append("ℹ️  Install scipy for full DCT analysis: pip install scipy")
            self._flags['dct_peak_to_avg'] = 0

        # 3. DWT Analysis (Discrete Wavelet Transform)
        if PYWT_AVAILABLE:
            dwt_results = self._analyze_dwt_watermarks(img_array)
            watermark_indicators.extend(dwt_results)
            # store detail entropy and unbalanced levels
            dwt_info = self.results.get('dwt_analysis', {})
            self._flags['dwt_detail_entropy'] = dwt_info.get('detail_entropy', 0)
            # detect unbalanced messages
            self._flags['dwt_unbalanced_levels'] = sum(1 for r in dwt_results if 'Unbalanced energy' in r)
        else:
            watermark_indicators.append("ℹ️  Install PyWavelets for DWT analysis: pip install pywavelets")
            self._flags['dwt_detail_entropy'] = 0
            self._flags['dwt_unbalanced_levels'] = 0

        # 4. Check for metadata watermark indicators
        if self.results.get('metadata'):
            watermark_tags = ['Watermark', 'Copyright', 'Rights', 'Attribution']
            for tag in watermark_tags:
                if tag in self.results['metadata']:
                    watermark_indicators.append(f"✓ {tag} in metadata: {self.results['metadata'][tag]}")

        self.results['watermark_indicators'] = watermark_indicators

        for indicator in watermark_indicators:
            print(f"  {indicator}")
        print()

    def calculate_hashes(self):
        """Calculate cryptographic hashes for image verification"""
        print("🔐 CRYPTOGRAPHIC HASHES")
        print("-" * 60)

        with open(self.image_path, 'rb') as f:
            file_bytes = f.read()

        hashes = {
            'md5': hashlib.md5(file_bytes).hexdigest(),
            'sha256': hashlib.sha256(file_bytes).hexdigest(),
        }

        self.results['hashes'] = hashes

        print(f"  MD5:    {hashes['md5']}")
        print(f"  SHA256: {hashes['sha256']}")
        print()

    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data (normalized to 0-1)"""
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return float((-np.sum(hist * np.log2(hist))) / 8)

    def _analyze_dct_watermarks(self, img_array):
        """Analyze DCT coefficients for watermark patterns"""
        results = []

        # Convert to grayscale for analysis
        gray = np.mean(img_array, axis=2).astype(np.float32)

        # Divide image into 8x8 blocks (JPEG uses 8x8 DCT blocks)
        block_size = 8
        h, w = gray.shape
        h_blocks = h // block_size
        w_blocks = w // block_size

        # Store mid-frequency coefficient statistics
        mid_freq_coeffs = []

        for i in range(h_blocks):
            for j in range(w_blocks):
                block = gray[i*block_size:(i+1)*block_size,
                           j*block_size:(j+1)*block_size]

                if block.shape == (block_size, block_size):
                    # Apply 2D DCT
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

                    # Extract mid-frequency coefficients (where watermarks are often embedded)
                    mid_freq = dct_block[2:6, 2:6].flatten()
                    mid_freq_coeffs.extend(mid_freq)

        mid_freq_coeffs = np.array(mid_freq_coeffs)

        # Analyze coefficient distribution
        coeff_std = np.std(mid_freq_coeffs)
        coeff_mean = np.mean(np.abs(mid_freq_coeffs))

        # Check for unusual patterns in DCT coefficients
        # Watermarks often create periodic patterns in frequency domain
        fft_of_dct = np.fft.fft(mid_freq_coeffs)
        power_spectrum = np.abs(fft_of_dct) ** 2

        # Look for strong periodic components (excluding DC)
        peak_power = np.max(power_spectrum[1:len(power_spectrum)//2]) if len(power_spectrum) > 2 else 0
        avg_power = np.mean(power_spectrum[1:len(power_spectrum)//2]) if len(power_spectrum) > 2 else 1

        ratio = (peak_power / avg_power) if avg_power > 0 else 0
        if ratio > 10:
            results.append(f"⚠️  DCT: Strong periodic pattern detected (ratio: {ratio:.1f})")
        else:
            results.append("✓ DCT: No strong periodic patterns")

        # Check coefficient statistics
        results.append(f"ℹ️  DCT mid-frequency std: {coeff_std:.2f}, mean: {coeff_mean:.2f}")

        # Store detailed results
        self.results['dct_analysis'] = {
            'mid_freq_std': float(coeff_std),
            'mid_freq_mean': float(coeff_mean),
            'peak_to_avg_ratio': float(ratio)
        }

        return results

    def _analyze_dwt_watermarks(self, img_array):
        """Analyze Discrete Wavelet Transform for watermark patterns"""
        results = []

        # Convert to grayscale
        gray = np.mean(img_array, axis=2)

        # Perform multi-level wavelet decomposition
        wavelet = 'haar'
        max_level = 3

        coeffs = pywt.wavedec2(gray, wavelet, level=max_level)

        # Analyze each level of decomposition
        for level in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level]  # Horizontal, Vertical, Diagonal details

            # Calculate statistics for each subband
            h_energy = np.sum(cH ** 2)
            v_energy = np.sum(cV ** 2)
            d_energy = np.sum(cD ** 2)

            total_energy = h_energy + v_energy + d_energy

            # Check for unusual energy distribution (watermarks affect this)
            if total_energy > 0:
                h_ratio = h_energy / total_energy
                v_ratio = v_energy / total_energy
                d_ratio = d_energy / total_energy

                # Highly unbalanced energy can indicate watermark
                max_ratio = max(h_ratio, v_ratio, d_ratio)
                if max_ratio > 0.6:
                    results.append(f"⚠️  DWT Level {level}: Unbalanced energy (max: {max_ratio:.2f})")

        # Check for patterns in wavelet coefficients
        cH, cV, cD = coeffs[1]  # First detail level

        # Calculate entropy of detail coefficients
        detail_entropy = self._calculate_entropy(cH.flatten())

        if detail_entropy > 0.95:
            results.append("⚠️  DWT: High entropy in detail coefficients")
        elif detail_entropy < 0.3:
            results.append("⚠️  DWT: Very low entropy (possible watermark)")
        else:
            results.append("✓ DWT: Normal coefficient distribution")

        # Store detailed results
        self.results['dwt_analysis'] = {
            'wavelet_used': wavelet,
            'levels': max_level,
            'detail_entropy': float(detail_entropy)
        }

        return results

    def compute_final_verdict(self):
        """Compute a final AI-vs-Real verdict and confidence score using heuristics."""
        # Start with a base score where higher means more likely AI
        score = 0.0
        reasons = []

        meta = self.results.get('metadata')
        basic = self.results.get('basic_info', {})
        stats = self.results.get('statistics', {})
        dct_ratio = float(self.results.get('dct_analysis', {}).get('peak_to_avg_ratio', 0))
        dwt_unbalanced = int(self._flags.get('dwt_unbalanced_levels', 0))

        # Metadata: presence of real camera metadata strongly reduces AI probability
        if meta is None:
            score += 30
            reasons.append('Missing EXIF metadata')
        else:
            # If metadata is present and contains realistic camera fields, decrease AI score
            if any(k in meta for k in ['Make', 'Model', 'DateTimeOriginal', 'Software']):
                score -= 40
                reasons.append('Valid EXIF metadata present')

        # DCT ratio: strong indicator
        if dct_ratio > 40:
            score += 50
            reasons.append(f'Very high DCT periodicity ({dct_ratio:.1f})')
        elif dct_ratio > 15:
            score += 30
            reasons.append(f'High DCT periodicity ({dct_ratio:.1f})')
        elif dct_ratio > 8:
            score += 10
            reasons.append(f'Moderate DCT periodicity ({dct_ratio:.1f})')
        else:
            score -= 10

        # DWT unbalanced levels
        if dwt_unbalanced >= 2:
            score += 20
            reasons.append(f'DWT unbalanced levels: {dwt_unbalanced}')
        elif dwt_unbalanced == 1:
            score += 8
            reasons.append('Some DWT unbalance')
        else:
            score -= 2

        # File format and size heuristics
        fmt = basic.get('format', '').upper()
        if fmt == 'PNG':
            score += 5
            reasons.append('PNG format (common for AI outputs)')

        fsize = basic.get('file_size_mb', 0)
        if fsize < 0.08:
            score += 10
            reasons.append('Very small file size (possible synthetic/compressed)')
        elif fsize > 3:
            score -= 5

        # Edge variance smoothness
        if self._flags.get('low_edge_variance'):
            score += 8
            reasons.append('Low edge variance (smoothness)')

        # Common AI aspect ratio
        if self._flags.get('common_ai_aspect_ratio'):
            score += 4

        # PNG 64-divisible
        if self._flags.get('png_64_divisible'):
            score += 5

        # High symmetry
        if self._flags.get('high_symmetry'):
            score += 6

        # Entropy checks: extremely low entropy might indicate synthetic flat images
        entropy = float(stats.get('entropy', 0))
        if entropy < 3.5:
            score += 5
            reasons.append('Low image entropy')
        elif entropy > 8.5:
            # extremely high entropy can be noisy/generated; small penalty
            score += 2

        # Unique colors / diversity heuristics
        color_div = float(stats.get('color_diversity_ratio', 0))
        if color_div > 0.25:
            # very high diversity sometimes occurs in generated textures
            score += 8
            reasons.append('Very high color diversity')
        elif color_div < 0.01:
            score += 5
            reasons.append('Very low color diversity')
        else:
            score -= 2

        # Clamp score and convert to 0-100 confidence for AI
        raw_score = max(0.0, min(100.0, score))

        # Map raw_score to confidence percentage for 'AI' label
        # We treat >=50 as "Likely AI", <50 as "Likely Real"
        confidence = raw_score
        is_ai = confidence >= 50

        verdict = 'AI-generated' if is_ai else 'Likely real photo'

        final = {
            'verdict': verdict,
            'is_ai': bool(is_ai),
            'confidence_percent': round(confidence, 1),
            'score_details': {
                'raw_score': round(raw_score, 2),
                'reasons': reasons
            }
        }

        self.results['final_verdict'] = final

        # Print final verdict succinctly
        print("🔎 FINAL VERDICT")
        print("-" * 60)
        print(f"  Verdict: {final['verdict']}")
        print(f"  Confidence (AI): {final['confidence_percent']}%")
        if reasons:
            print("  Top reasons:")
            for r in reasons[:5]:
                print(f"    - {r}")
        print()

    def export_results(self, output_path=None):
        """Export results to JSON file"""
        if output_path is None:
            output_path = f"{Path(self.image_path).stem}_analysis.json"

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"✓ Results exported to: {output_path}\n")


def main():
    # Prompt user for image path
    image_path = input('Enter path to image file: ').strip().strip('"')

    if not image_path:
        print('No image path provided. Exiting.')
        return

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    try:
        analyzer = ImageForensicsAnalyzer(image_path)
        analyzer.analyze_all()

        # Ask if user wants to export results
        export = input("Export results to JSON? (y/n): ").lower()
        if export == 'y':
            output = input('Output JSON filename (or press Enter to use default): ').strip()
            analyzer.export_results(output if output else None)

    except Exception as e:
        print(f"Error analyzing image: {e}")


if __name__ == "__main__":
    main()
