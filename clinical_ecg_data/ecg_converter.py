import wfdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import logging
from datetime import datetime
import gc
import scipy.signal  
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
import concurrent.futures

# Configure non-interactive backend first
matplotlib.use('Agg')  # Set before other matplotlib imports
plt.ioff()  # Disable interactive mode

# Configure clinical logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CLINICAL ECG - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_conversion.log'),
        logging.StreamHandler()
    ]
)

# Updated medical device constants with filtering parameters
CLINICAL_STANDARDS = {
    'SAMPLING_RATE': 500,
    'RESOLUTION': 100,  # 100 DPI for 512px = 5.12 inches
    'MM_PER_MV': 10,
    'LEAD_ORDER': [
        'I', 'II', 'III',
        'aVR', 'aVL', 'aVF',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ],
    'SIGNAL_RANGE': (-5, 5),
    'TRACE_WIDTH': 0.8,
    'LEAD_SPACING': 25,
    'PAGE_SIZE': (512, 512),  
    'FILTERS': {
        'HIGH_PASS': 0.5,   # Hz (baseline wander removal)
        'LOW_PASS': 150,    # Hz (anti-aliasing)
        'NOTCH': 50         # Hz (powerline interference)
    },
    'VALID_EXTENSIONS' : ('.dat', '.hea'),
    'OUTPUT_FORMAT' : 'PNG'
}



class ECGFigureManager:
    """Context manager for figure lifecycle management"""
    """Context manager for figure lifecycle management"""
    def __init__(self, record):
        self.record = record
        self.fig = None
        self.img = None
        
        
    def __enter__(self):
        """Create and configure figure"""
        # Calculate figure size in inches for exact 512x512 pixels
        dpi = CLINICAL_STANDARDS['RESOLUTION']
        figsize = (
            CLINICAL_STANDARDS['PAGE_SIZE'][0]/dpi,
            CLINICAL_STANDARDS['PAGE_SIZE'][1]/dpi
        )
        
        self.fig = plt.figure(
            figsize=figsize,
            dpi=dpi,
            facecolor='black'
        )
        self._create_plots()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup resources when exiting the context"""
        self._cleanup()

        
    def _apply_clinical_filters(self, signal):
        """IEC 60601-2-25 compliant signal conditioning"""
        fs = CLINICAL_STANDARDS['SAMPLING_RATE']
        nyquist = 0.5 * fs
        
        # High-pass filter for baseline wander removal
        b, a = scipy.signal.butter(
            2, 
            CLINICAL_STANDARDS['FILTERS']['HIGH_PASS']/nyquist, 
            btype='high'
        )
        filtered = scipy.signal.filtfilt(b, a, signal)
        
        # Low-pass anti-aliasing filter
        b, a = scipy.signal.butter(
            2, 
            CLINICAL_STANDARDS['FILTERS']['LOW_PASS']/nyquist, 
            btype='low'
        )
        filtered = scipy.signal.filtfilt(b, a, filtered)
        
        # Notch filter for powerline interference
        if CLINICAL_STANDARDS['FILTERS']['NOTCH']:
            b, a = scipy.signal.iirnotch(
                CLINICAL_STANDARDS['FILTERS']['NOTCH'],
                Q=30,
                fs=fs
            )
            filtered = scipy.signal.filtfilt(b, a, filtered)
            
        return filtered
    
    
    def _create_plots(self):
        """Create plots with tight layout"""
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('black')
        
        # Tight layout parameters
        self.fig.subplots_adjust(
            left=0,          # Remove left margin
            right=1,          # Remove right margin
            bottom=0,         # Remove bottom margin
            top=1,            # Remove top margin
            wspace=0,         # No horizontal space between subplots
            hspace=0          # No vertical space between subplots
        )
        
        # Get filtered signals matrix [12 leads x samples]
        raw_signals = self.record.p_signal.T
        filtered_signals = np.apply_along_axis(
            self._apply_clinical_filters, 
            1, 
            raw_signals
        )
        
        # Calculate vertical offsets
        offsets = np.arange(11, -1, -1) * CLINICAL_STANDARDS['LEAD_SPACING']
        
        # Vectorized signal processing
        baselines = np.mean(filtered_signals, axis=1, keepdims=True)
        calibrated = (filtered_signals - baselines) * CLINICAL_STANDARDS['MM_PER_MV']
        scaled_signals = calibrated + offsets[:, np.newaxis]
        
        # Single vectorized plot command
        ax.plot(
            scaled_signals.T,  # Transpose to [samples x 12 leads]
            color='white',
            linewidth=CLINICAL_STANDARDS['TRACE_WIDTH'],
            solid_capstyle='round',
            antialiased=True
        )
        
        # Set clinical display limits
        ax.set_xlim(0, filtered_signals.shape[1])
        ax.set_ylim(
            -CLINICAL_STANDARDS['LEAD_SPACING']/2,
            offsets[0] + CLINICAL_STANDARDS['LEAD_SPACING']/2
        )
        ax.axis('off')
    
    def render_to_image(self):
        """Convert figure to cropped medical-grade image"""
        if self.fig:
            self.fig.canvas.draw()
            
            # Get RGB buffer and convert to grayscale
            img_array = np.frombuffer(
                self.fig.canvas.tostring_rgb(), dtype=np.uint8
            ).reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            
            self.img = Image.fromarray(img_array).convert('L')
            
            # Clinical resize with anti-aliasing
            self.img = self.img.resize(
                CLINICAL_STANDARDS['PAGE_SIZE'],
                resample=Image.LANCZOS
            )
            
            # Remove borders using content-aware crop
            self.img = ImageOps.crop(self.img, border=0)
            self.img = ImageOps.autocontrast(self.img)
            
            self._cleanup()
            return self.img
        return None
    

    def _cleanup(self):
        """Resource cleanup"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
        gc.collect()

def validate_clinical_ecg(record):
    """Medical validation checks"""
    checks = {
        'lead_count': (record.p_signal.shape[1] == 12,
                      f"Invalid lead count: {record.p_signal.shape[1]}"),
        'sampling_rate': (record.fs == CLINICAL_STANDARDS['SAMPLING_RATE'],
                         f"Sampling rate {record.fs}Hz != 500Hz"),
        'signal_quality': (not np.isnan(record.p_signal).any(),
                         "ECG contains NaN values"),
        'voltage_range': (record.p_signal.min() > CLINICAL_STANDARDS['SIGNAL_RANGE'][0] and
                          record.p_signal.max() < CLINICAL_STANDARDS['SIGNAL_RANGE'][1],
                         "Signal out of clinical range (-5mV to +5mV)")
    }
    
    for check_name, (status, msg) in checks.items():
        if not status:
            logging.error(f"Clinical validation failed - {msg}")
            raise ValueError(f"Clinical standard violation: {msg}")

def process_ecg_record(record):
    """Process ECG record with memory-safe operations"""
    with ECGFigureManager(record) as manager:
        ecg_image = manager.render_to_image()
        if ecg_image is None:
            raise RuntimeError("Failed to generate ECG image")
        return ecg_image

def save_medical_image(img, output_path):
    """Save with medical standards (PNG format)"""
    try:
        img.save(
            output_path,
            format=CLINICAL_STANDARDS['OUTPUT_FORMAT'],
            compress_level=9,        # Max compression
            dpi=(CLINICAL_STANDARDS['RESOLUTION'], 
                 CLINICAL_STANDARDS['RESOLUTION']),
            optimize=True
        )
        logging.info(f"Saved clinical ECG to {output_path}")
    finally:
        if img:
            img.close()

def process_single_record(record_id, input_dir, output_dir):
    """Process single ECG record with error handling"""
    try:
        record_path = os.path.join(input_dir, record_id)
        record = wfdb.rdrecord(record_path)
        validate_clinical_ecg(record)
        
        with ECGFigureManager(record) as manager:
            ecg_img = manager.render_to_image()
            if ecg_img is None:
                raise RuntimeError("Image generation failed")


            output_id =  record_id = record_id.replace('_hr', '')
            output_path = os.path.join(output_dir, f"ECG{int(output_id):05d}_clinical_512.png")
            save_medical_image(ecg_img, output_path)
            
        return True, record_id
    except Exception as e:
        logging.error(f"Failed {record_id}: {str(e)}")
        return False, record_id

def find_ecg_records(input_dir):
    """Find valid ECG records in directory"""
    records = set()
    for fname in os.listdir(input_dir):
        if fname.endswith('.hea'):
            base_name = fname[:-4]
            dat_file = f"{base_name}.dat"
            if os.path.exists(os.path.join(input_dir, dat_file)):
                records.add(base_name)
    return sorted(records)

def process_batch(input_dir, output_dir, workers=4):
    """Batch process all ECG records in directory"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    record_ids = find_ecg_records(input_dir)
    
    if not record_ids:
        logging.warning(f"No valid ECG records found in {input_dir}")
        return
    
    logging.info(f"Found {len(record_ids)} ECG records to process")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_single_record,
                record_id,
                input_dir,
                output_dir
            )
            for record_id in record_ids
        ]
        
        with tqdm(total=len(futures), desc="Processing ECG Records") as pbar:
            for future in concurrent.futures.as_completed(futures):
                success, record_id = future.result()
                if success:
                    success_count += 1
                pbar.update(1)
                pbar.set_postfix_str(f"Last processed: {record_id}")
    
    logging.info(f"Batch processing complete. Success: {success_count}/"
                 f"{len(record_ids)} ({success_count/len(record_ids):.1%})")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert ECG records to medical images')
    parser.add_argument('-i', '--input', required=True,
                       help='Input directory with ECG records')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for images')
    parser.add_argument('-w', '--workers', type=int, default=4,
                       help='Number of parallel workers')
    args = parser.parse_args()
    
    try:
        start_time = datetime.now()
        logging.info(f"Starting batch processing from {args.input}")
        
        process_batch(args.input, args.output, args.workers)
        
        duration = datetime.now() - start_time
        logging.info(f"Total processing time: {duration.total_seconds():.2f}s")
        
    except Exception as e:
        logging.critical(f"Fatal error in batch processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
