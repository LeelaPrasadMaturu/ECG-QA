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

# Configure non-interactive backend
matplotlib.use('Agg')  # Set before other matplotlib imports
plt.ioff()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CLINICAL ECG - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_conversion.log'),
        logging.StreamHandler()
    ]
)

CLINICAL_STANDARDS = {
    'SAMPLING_RATE': 500,
    'RESOLUTION': 100,  
    'MM_PER_MV': 9,
    'LEAD_ORDER': [
        'I', 'II', 'III',   
        'aVR', 'aVL', 'aVF',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ],
    'SIGNAL_RANGE': (-5, 5),
    'TRACE_WIDTH': 0.8,
    'LEAD_SPACING': 30,
    'PAGE_SIZE': (512, 512),
    'FILTERS': {
        'HIGH_PASS': 0.5,
        'LOW_PASS': 150,
        'NOTCH': 50
    },
    'VALID_EXTENSIONS': ('.dat', '.hea'),
    'OUTPUT_FORMAT': 'PNG'
}

class ECGFigureManager:
    def __init__(self, record):
        self.record = record
        self.fig = None
        self.img = None

    def __enter__(self):
        dpi = CLINICAL_STANDARDS['RESOLUTION']
        figsize = (
            CLINICAL_STANDARDS['PAGE_SIZE'][0] / dpi,
            CLINICAL_STANDARDS['PAGE_SIZE'][1] / dpi
        )
        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='black')
        self._create_plots()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._cleanup()

    def _apply_clinical_filters(self, signal):
        fs = CLINICAL_STANDARDS['SAMPLING_RATE']
        nyquist = 0.5 * fs

        b, a = scipy.signal.butter(2, CLINICAL_STANDARDS['FILTERS']['HIGH_PASS'] / nyquist, btype='high')
        filtered = scipy.signal.filtfilt(b, a, signal)

        b, a = scipy.signal.butter(2, CLINICAL_STANDARDS['FILTERS']['LOW_PASS'] / nyquist, btype='low')
        filtered = scipy.signal.filtfilt(b, a, filtered)

        if CLINICAL_STANDARDS['FILTERS']['NOTCH']:
            b, a = scipy.signal.iirnotch(
                CLINICAL_STANDARDS['FILTERS']['NOTCH'], Q=30, fs=fs
            )
            filtered = scipy.signal.filtfilt(b, a, filtered)

        return filtered

    def _create_plots(self):
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('black')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        raw_signals = self.record.p_signal.T
        filtered_signals = np.apply_along_axis(self._apply_clinical_filters, 1, raw_signals)

        total_height = CLINICAL_STANDARDS['PAGE_SIZE'][1]
        top_bottom_margin = 3
        gap = 2
        n_leads = 12
        available_height = total_height - 2 * top_bottom_margin
        total_gap_space = (n_leads - 1) * gap
        usable_height = available_height - total_gap_space
        lead_height = usable_height / n_leads

        y_offsets = np.array([
            total_height - (top_bottom_margin + i * (lead_height + gap) + lead_height / 2)
            for i in range(n_leads)
        ])

        baselines = np.mean(filtered_signals, axis=1, keepdims=True)
        calibrated = (filtered_signals - baselines) * CLINICAL_STANDARDS['MM_PER_MV']
        scaled_signals = calibrated + y_offsets[:, np.newaxis]

        ax.plot(
            scaled_signals.T,
            color='white',
            linewidth=CLINICAL_STANDARDS['TRACE_WIDTH'],
            solid_capstyle='round',
            antialiased=True
        )

        ax.set_xlim(0, filtered_signals.shape[1])
        ax.set_ylim(0, total_height)
        ax.axis('off')

    def render_to_image(self):
        if self.fig:
            self.fig.canvas.draw()
            img_array = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.img = Image.fromarray(255 - img_array).convert('L')
            self.img = self.img.resize(CLINICAL_STANDARDS['PAGE_SIZE'], resample=Image.LANCZOS)
            self.img = ImageOps.crop(self.img, border=0)
            self.img = ImageOps.autocontrast(self.img)
            self._cleanup()
            return self.img
        return None

    def _cleanup(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
        gc.collect()

def validate_clinical_ecg(record):
    checks = {
        'lead_count': (record.p_signal.shape[1] == 12, f"Invalid lead count: {record.p_signal.shape[1]}"),
        'sampling_rate': (record.fs == CLINICAL_STANDARDS['SAMPLING_RATE'], f"Sampling rate {record.fs}Hz != 500Hz"),
        'signal_quality': (not np.isnan(record.p_signal).any(), "ECG contains NaN values"),
        'voltage_range': (
            record.p_signal.min() > CLINICAL_STANDARDS['SIGNAL_RANGE'][0] and
            record.p_signal.max() < CLINICAL_STANDARDS['SIGNAL_RANGE'][1],
            "Signal out of clinical range (-5mV to +5mV)"
        )
    }
    for _, (status, msg) in checks.items():
        if not status:
            logging.error(f"Clinical validation failed - {msg}")
            raise ValueError(f"Clinical standard violation: {msg}")

def process_ecg_record(record):
    with ECGFigureManager(record) as manager:
        ecg_image = manager.render_to_image()
        if ecg_image is None:
            raise RuntimeError("Failed to generate ECG image")
        return ecg_image

def save_medical_image(img, output_path):
    try:
        img.save(
            output_path,
            format=CLINICAL_STANDARDS['OUTPUT_FORMAT'],
            compress_level=9,
            dpi=(CLINICAL_STANDARDS['RESOLUTION'], CLINICAL_STANDARDS['RESOLUTION']),
            optimize=True
        )
        logging.info(f"Saved clinical ECG to {output_path}")
    finally:
        if img:
            img.close()

def process_single_record(record_id, input_dir, output_dir):
    try:
        record_path = os.path.join(input_dir, record_id)
        record = wfdb.rdrecord(record_path)
        validate_clinical_ecg(record)

        with ECGFigureManager(record) as manager:
            ecg_img = manager.render_to_image()
            if ecg_img is None:
                raise RuntimeError("Image generation failed")

            output_id = record_id.replace('_hr', '')
            output_path = os.path.join(output_dir, f"ECG{int(output_id):05d}_clinical_512.png")
            save_medical_image(ecg_img, output_path)

        return True, record_id
    except Exception as e:
        logging.error(f"Failed {record_id}: {str(e)}")
        return False, record_id

def find_ecg_records(input_dir):
    records = set()
    for fname in os.listdir(input_dir):
        if fname.endswith('.hea'):
            base_name = fname[:-4]
            dat_file = f"{base_name}.dat"
            if os.path.exists(os.path.join(input_dir, dat_file)):
                records.add(base_name)
    return sorted(records)

def process_batch(input_dir, output_dir, workers=4):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    record_ids = find_ecg_records(input_dir)

    if not record_ids:
        logging.warning(f"No valid ECG records found in {input_dir}")
        return

    logging.info(f"Found {len(record_ids)} ECG records to process")

    success_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_single_record, record_id, input_dir, output_dir)
            for record_id in record_ids
        ]
        with tqdm(total=len(futures), desc="Processing ECG Records") as pbar:
            for future in concurrent.futures.as_completed(futures):
                success, record_id = future.result()
                if success:
                    success_count += 1
                pbar.update(1)
                pbar.set_postfix_str(f"Last processed: {record_id}")

    logging.info(f"Batch processing complete. Success: {success_count}/{len(record_ids)} ({success_count/len(record_ids):.1%})")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert ECG records to medical images')
    parser.add_argument('-i', '--input', required=True, help='Input directory with ECG records')
    parser.add_argument('-o', '--output', required=True, help='Output directory for images')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Number of parallel workers')
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