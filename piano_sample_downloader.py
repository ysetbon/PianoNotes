#!/usr/bin/env python3
"""
Piano Sample Downloader and Organizer
Downloads piano samples from multiple sources and organizes them by pitch (A0-C8)
"""

import os
import re
import json
import time
import shutil
import logging
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PitchMapper:
    """Maps between different pitch representations (MIDI, scientific notation, etc.)"""

    # Piano range: A0 (MIDI 21) to C8 (MIDI 108)
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    # Alternative naming conventions
    ALT_SHARP = {'C#': 'Cs', 'D#': 'Ds', 'F#': 'Fs', 'G#': 'Gs', 'A#': 'As'}
    ALT_FLAT = {'Db': 'Df', 'Eb': 'Ef', 'Gb': 'Gf', 'Ab': 'Af', 'Bb': 'Bf'}

    @classmethod
    def midi_to_note(cls, midi_num: int) -> str:
        """Convert MIDI number to scientific pitch notation (e.g., 21 -> 'A0')"""
        if midi_num < 21 or midi_num > 108:
            return None
        octave = (midi_num - 12) // 12
        note_idx = midi_num % 12
        return f"{cls.NOTE_NAMES[note_idx]}{octave}"

    @classmethod
    def note_to_midi(cls, note: str) -> Optional[int]:
        """Convert scientific pitch notation to MIDI number (e.g., 'A0' -> 21)"""
        # Parse note name and octave
        match = re.match(r'^([A-Ga-g])([#b]?)(-?\d+)$', note.strip())
        if not match:
            return None

        note_letter = match.group(1).upper()
        accidental = match.group(2)
        octave = int(match.group(3))

        # Find base note index
        note_idx = None
        for i, name in enumerate(cls.NOTE_NAMES):
            if name[0] == note_letter:
                note_idx = i
                break

        if note_idx is None:
            return None

        # Apply accidental
        if accidental == '#':
            note_idx += 1
        elif accidental == 'b':
            note_idx -= 1

        # Handle wraparound
        note_idx = note_idx % 12

        # Calculate MIDI number
        midi_num = (octave + 1) * 12 + note_idx

        if 21 <= midi_num <= 108:
            return midi_num
        return None

    @classmethod
    def get_all_pitches(cls) -> List[str]:
        """Get list of all piano pitches from A0 to C8"""
        pitches = []
        for midi_num in range(21, 109):  # A0 (21) to C8 (108)
            pitches.append(cls.midi_to_note(midi_num))
        return pitches

    @classmethod
    def get_folder_name(cls, pitch: str) -> str:
        """Get folder-safe name for pitch (replace # with 'sharp')"""
        return pitch.replace('#', 'sharp')

    @classmethod
    def parse_filename_for_pitch(cls, filename: str) -> Optional[str]:
        """
        Try to extract pitch from filename using various naming conventions.
        Returns normalized pitch name or None.
        """
        filename_lower = filename.lower()
        filename_upper = filename.upper()

        # Pattern 1: Scientific notation (A0, C#4, Bb3, etc.)
        patterns = [
            r'[_\-\s]([A-Ga-g][#b]?\d)[_\-\s\.]',  # _A0_ or -C#4.
            r'^([A-Ga-g][#b]?\d)[_\-\s\.]',        # A0_ at start
            r'[_\-\s]([A-Ga-g][#b]?\d)$',          # _A0 at end
            r'[_\-\s]([A-Ga-g]s?\d)[_\-\s\.]',     # Cs4 format (s for sharp)
            r'[_\-\s]([A-Ga-g]f?\d)[_\-\s\.]',     # Df4 format (f for flat)
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                note_str = match.group(1)
                # Handle 's' for sharp and 'f' for flat
                note_str = re.sub(r'([A-Ga-g])s(\d)', r'\1#\2', note_str, flags=re.IGNORECASE)
                note_str = re.sub(r'([A-Ga-g])f(\d)', r'\1b\2', note_str, flags=re.IGNORECASE)
                midi = cls.note_to_midi(note_str)
                if midi:
                    return cls.midi_to_note(midi)

        # Pattern 2: MIDI number (e.g., _021.wav, _21.wav, note21.wav)
        midi_patterns = [
            r'[_\-\s]0?(\d{1,3})[_\-\s\.]',  # _021. or _21.
            r'note[_\-]?0?(\d{1,3})',         # note21 or note_21
            r'midi[_\-]?0?(\d{1,3})',         # midi21
        ]

        for pattern in midi_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                midi_num = int(match.group(1))
                if 21 <= midi_num <= 108:
                    return cls.midi_to_note(midi_num)

        # Pattern 3: Key number on 88-key piano (Key 1 = A0)
        key_patterns = [
            r'key[_\-\s]?(\d{1,2})',
            r'k(\d{1,2})[_\-\s\.]',
        ]

        for pattern in key_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                key_num = int(match.group(1))
                if 1 <= key_num <= 88:
                    midi_num = key_num + 20  # Key 1 = MIDI 21 (A0)
                    return cls.midi_to_note(midi_num)

        return None


class BaseDownloader:
    """Base class for sample downloaders"""

    def __init__(self, output_dir: str, session: requests.Session = None):
        self.output_dir = Path(output_dir)
        self.session = session or requests.Session()
        self.session.headers.update({
            'User-Agent': 'PianoSampleCollector/1.0 (Research Project)'
        })

    def ensure_pitch_folders(self):
        """Create all pitch folders"""
        for pitch in PitchMapper.get_all_pitches():
            folder_name = PitchMapper.get_folder_name(pitch)
            folder_path = self.output_dir / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)

    def save_sample(self, data: bytes, pitch: str, source: str,
                    original_filename: str, velocity: str = None) -> Path:
        """Save a sample to the appropriate pitch folder"""
        folder_name = PitchMapper.get_folder_name(pitch)
        folder_path = self.output_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        # Create unique filename
        ext = Path(original_filename).suffix or '.wav'
        velocity_str = f"_v{velocity}" if velocity else ""
        # Create hash of content for uniqueness
        content_hash = hashlib.md5(data).hexdigest()[:8]
        filename = f"{source}_{pitch}{velocity_str}_{content_hash}{ext}"

        file_path = folder_path / filename
        with open(file_path, 'wb') as f:
            f.write(data)

        logger.info(f"Saved: {file_path}")
        return file_path

    def download_file(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download a file with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                return response.content
            except requests.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None


class NSynthDownloader(BaseDownloader):
    """Download samples from NSynth dataset"""

    NSYNTH_URLS = {
        'train': 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz',
        'valid': 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz',
        'test': 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz',
    }

    # Direct TFRecord access is complex, so we'll use the JSON index approach
    NSYNTH_JSON_URL = 'https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-test/examples.json'

    def __init__(self, output_dir: str, session: requests.Session = None):
        super().__init__(output_dir, session)
        self.source_name = "nsynth"

    def download_metadata(self) -> Dict:
        """Download NSynth metadata JSON"""
        logger.info("Downloading NSynth metadata...")
        try:
            # Try test set first (smaller)
            response = self.session.get(self.NSYNTH_JSON_URL, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to download NSynth metadata: {e}")
            return {}

    def filter_piano_samples(self, metadata: Dict) -> List[Dict]:
        """Filter for keyboard/piano samples"""
        piano_samples = []
        for sample_id, info in metadata.items():
            # Filter for keyboard family
            if info.get('instrument_family_str') == 'keyboard' or \
               info.get('instrument_family') == 3:  # 3 = keyboard
                # Check pitch range (21-108 for piano)
                pitch = info.get('pitch', 0)
                if 21 <= pitch <= 108:
                    info['sample_id'] = sample_id
                    piano_samples.append(info)
        return piano_samples

    def get_download_instructions(self) -> str:
        """Return instructions for manual NSynth download"""
        return """
NSynth Dataset Download Instructions:
=====================================
The NSynth dataset is large (several GB). To download:

1. Visit: https://magenta.withgoogle.com/datasets/nsynth
2. Download the test set (smallest): nsynth-test.jsonwav.tar.gz
3. Extract to a folder
4. Run this script with --nsynth-path /path/to/extracted/folder

Alternative - Use TensorFlow Datasets:
    pip install tensorflow-datasets
    import tensorflow_datasets as tfds
    ds = tfds.load('nsynth', split='test')
"""


class FreesoundDownloader(BaseDownloader):
    """Download samples from Freesound.org"""

    API_BASE = "https://freesound.org/apiv2"

    def __init__(self, output_dir: str, api_key: str = None, session: requests.Session = None):
        super().__init__(output_dir, session)
        self.api_key = api_key
        self.source_name = "freesound"

    def search_piano_samples(self, query: str = "piano note single",
                             page_size: int = 50, max_pages: int = 10) -> List[Dict]:
        """Search for piano samples on Freesound"""
        if not self.api_key:
            logger.warning("Freesound API key not provided. Skipping Freesound downloads.")
            return []

        samples = []
        for page in range(1, max_pages + 1):
            try:
                params = {
                    'query': query,
                    'token': self.api_key,
                    'page_size': page_size,
                    'page': page,
                    'fields': 'id,name,duration,download,previews,tags',
                    'filter': 'duration:[1 TO 30]',  # 1-30 seconds
                }
                response = self.session.get(f"{self.API_BASE}/search/text/", params=params)
                response.raise_for_status()
                data = response.json()

                results = data.get('results', [])
                if not results:
                    break

                samples.extend(results)
                logger.info(f"Freesound: Retrieved page {page}, total samples: {len(samples)}")

                if not data.get('next'):
                    break

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Freesound search error: {e}")
                break

        return samples

    def download_sample(self, sample_info: Dict) -> Optional[Tuple[bytes, str]]:
        """Download a sample from Freesound"""
        if not self.api_key:
            return None

        try:
            # Use preview URL (doesn't require OAuth)
            preview_url = sample_info.get('previews', {}).get('preview-hq-mp3')
            if not preview_url:
                preview_url = sample_info.get('previews', {}).get('preview-lq-mp3')

            if preview_url:
                data = self.download_file(preview_url)
                if data:
                    return data, sample_info.get('name', 'unknown')
        except Exception as e:
            logger.error(f"Failed to download Freesound sample {sample_info.get('id')}: {e}")

        return None

    def get_api_instructions(self) -> str:
        """Return instructions for getting Freesound API key"""
        return """
Freesound API Setup:
====================
1. Create an account at https://freesound.org/
2. Go to https://freesound.org/apiv2/apply/
3. Create an application to get your API key
4. Run this script with --freesound-key YOUR_API_KEY
"""


class PianobookDownloader(BaseDownloader):
    """Download and extract samples from Pianobook"""

    PIANOBOOK_BASE = "https://www.pianobook.co.uk"
    PACKS_URL = "https://www.pianobook.co.uk/packs/"

    def __init__(self, output_dir: str, session: requests.Session = None):
        super().__init__(output_dir, session)
        self.source_name = "pianobook"

    def get_pack_list(self) -> List[Dict]:
        """
        Get list of available sample packs from Pianobook.
        Note: Pianobook doesn't have a public API, so this provides
        a curated list of known good packs with direct links.
        """
        # Curated list of known Pianobook packs with sample formats
        # These are well-documented packs that expose WAV samples
        known_packs = [
            {
                'name': 'The Experience',
                'url': 'https://www.pianobook.co.uk/packs/the-experience/',
                'description': 'Steinway Model B - High quality multi-velocity'
            },
            {
                'name': 'Soft Piano',
                'url': 'https://www.pianobook.co.uk/packs/soft-piano/',
                'description': 'Soft, intimate piano samples'
            },
            {
                'name': 'The Grand Toy Piano',
                'url': 'https://www.pianobook.co.uk/packs/the-grand-toy-piano/',
                'description': 'Toy piano samples'
            },
            {
                'name': 'Upright Piano',
                'url': 'https://www.pianobook.co.uk/packs/upright-piano/',
                'description': 'Classic upright piano'
            },
        ]
        return known_packs

    def get_download_instructions(self) -> str:
        """Return instructions for Pianobook downloads"""
        return """
Pianobook Manual Download Instructions:
=======================================
Pianobook requires manual download due to their community model.

1. Visit https://www.pianobook.co.uk/packs/
2. Filter by "Pianos" -> "Acoustic"
3. Download packs in "Decent Sampler" or "SFZ" format (these expose raw WAVs)
4. Extract the ZIP files
5. Run this script with --pianobook-path /path/to/extracted/packs/

The script will scan for WAV files and organize them by detected pitch.

Recommended packs for comprehensive coverage:
- The Experience (Steinway Model B)
- Soft Piano
- Upright felt pianos
- Any pack with "multi-sample" or "chromatic" in description
"""


class LocalSampleOrganizer(BaseDownloader):
    """Organize locally downloaded samples into pitch folders"""

    AUDIO_EXTENSIONS = {'.wav', '.aiff', '.aif', '.flac', '.mp3', '.ogg'}

    def __init__(self, output_dir: str, session: requests.Session = None):
        super().__init__(output_dir, session)

    def scan_and_organize(self, source_dir: str, source_name: str = "local") -> Dict[str, int]:
        """
        Scan a directory for audio files and organize by detected pitch.
        Returns count of files organized per pitch.
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return {}

        organized = {}

        # Walk through all files
        for file_path in source_path.rglob('*'):
            if file_path.suffix.lower() not in self.AUDIO_EXTENSIONS:
                continue

            # Try to detect pitch from filename
            pitch = PitchMapper.parse_filename_for_pitch(file_path.name)

            if pitch:
                # Copy file to appropriate folder
                folder_name = PitchMapper.get_folder_name(pitch)
                dest_folder = self.output_dir / folder_name
                dest_folder.mkdir(parents=True, exist_ok=True)

                # Create unique filename
                content_hash = hashlib.md5(file_path.read_bytes()).hexdigest()[:8]
                new_filename = f"{source_name}_{pitch}_{content_hash}{file_path.suffix}"
                dest_path = dest_folder / new_filename

                if not dest_path.exists():
                    shutil.copy2(file_path, dest_path)
                    organized[pitch] = organized.get(pitch, 0) + 1
                    logger.info(f"Organized: {file_path.name} -> {pitch}/")
            else:
                logger.debug(f"Could not detect pitch from: {file_path.name}")

        return organized

    def process_decent_sampler_preset(self, dspreset_path: str) -> List[Tuple[str, str]]:
        """
        Parse a Decent Sampler .dspreset file to extract sample mappings.
        Returns list of (sample_path, pitch) tuples.
        """
        import xml.etree.ElementTree as ET

        mappings = []
        try:
            tree = ET.parse(dspreset_path)
            root = tree.getroot()

            # Find all sample elements
            for sample in root.iter('sample'):
                path = sample.get('path')
                root_note = sample.get('rootNote')

                if path and root_note:
                    try:
                        midi_num = int(root_note)
                        pitch = PitchMapper.midi_to_note(midi_num)
                        if pitch:
                            mappings.append((path, pitch))
                    except ValueError:
                        # rootNote might be note name
                        pitch = PitchMapper.note_to_midi(root_note)
                        if pitch:
                            mappings.append((path, PitchMapper.midi_to_note(pitch)))

        except Exception as e:
            logger.error(f"Error parsing Decent Sampler preset: {e}")

        return mappings

    def process_sfz_file(self, sfz_path: str) -> List[Tuple[str, str]]:
        """
        Parse an SFZ file to extract sample mappings.
        Returns list of (sample_path, pitch) tuples.
        """
        mappings = []
        sfz_dir = Path(sfz_path).parent

        try:
            with open(sfz_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            current_sample = None
            current_key = None

            for line in content.split('\n'):
                line = line.strip()

                # Look for sample= and key=/pitch_keycenter=
                if 'sample=' in line.lower():
                    match = re.search(r'sample=([^\s]+)', line, re.IGNORECASE)
                    if match:
                        current_sample = match.group(1)

                # Various key assignment opcodes
                for opcode in ['key=', 'pitch_keycenter=', 'lokey=']:
                    if opcode in line.lower():
                        match = re.search(rf'{opcode}(\d+|[a-g][#b]?\d)', line, re.IGNORECASE)
                        if match:
                            key_val = match.group(1)
                            try:
                                midi_num = int(key_val)
                            except ValueError:
                                midi_num = PitchMapper.note_to_midi(key_val)

                            if midi_num and 21 <= midi_num <= 108:
                                current_key = PitchMapper.midi_to_note(midi_num)

                # If we have both sample and key, add mapping
                if current_sample and current_key:
                    mappings.append((current_sample, current_key))
                    current_sample = None
                    current_key = None

        except Exception as e:
            logger.error(f"Error parsing SFZ file: {e}")

        return mappings


class UniversityOfIowaDownloader(BaseDownloader):
    """Download samples from University of Iowa MIS"""

    # The actual samples are at this mirror/archive location
    BASE_URL = "http://theremin.music.uiowa.edu/MISpiano.html"

    def __init__(self, output_dir: str, session: requests.Session = None):
        super().__init__(output_dir, session)
        self.source_name = "iowa"

    def get_download_instructions(self) -> str:
        """Return instructions for Iowa samples"""
        return """
University of Iowa Musical Instrument Samples:
==============================================
1. Visit: http://theremin.music.uiowa.edu/MIS.html
2. Navigate to Piano section
3. Download the Steinway samples
4. Extract and run with --iowa-path /path/to/samples/

Alternative mirror: https://archive.org/details/UniversityOfIowaMIS
"""


class PianoSampleCollector:
    """Main orchestrator for collecting piano samples from all sources"""

    def __init__(self, output_dir: str = "notes",
                 freesound_key: str = None,
                 nsynth_path: str = None,
                 pianobook_path: str = None,
                 iowa_path: str = None):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PianoSampleCollector/1.0 (Research/Educational)'
        })

        # Initialize downloaders
        self.nsynth = NSynthDownloader(output_dir, self.session)
        self.freesound = FreesoundDownloader(output_dir, freesound_key, self.session)
        self.pianobook = PianobookDownloader(output_dir, self.session)
        self.iowa = UniversityOfIowaDownloader(output_dir, self.session)
        self.organizer = LocalSampleOrganizer(output_dir, self.session)

        # Local paths for manual downloads
        self.nsynth_path = nsynth_path
        self.pianobook_path = pianobook_path
        self.iowa_path = iowa_path

    def setup_folders(self):
        """Create all pitch folders"""
        logger.info("Setting up pitch folders...")
        for pitch in PitchMapper.get_all_pitches():
            folder_name = PitchMapper.get_folder_name(pitch)
            folder_path = self.output_dir / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created {len(PitchMapper.get_all_pitches())} pitch folders in {self.output_dir}")

    def process_local_nsynth(self) -> Dict[str, int]:
        """Process locally downloaded NSynth samples"""
        if not self.nsynth_path:
            return {}

        logger.info(f"Processing NSynth samples from: {self.nsynth_path}")
        return self.organizer.scan_and_organize(self.nsynth_path, "nsynth")

    def process_local_pianobook(self) -> Dict[str, int]:
        """Process locally downloaded Pianobook samples"""
        if not self.pianobook_path:
            return {}

        logger.info(f"Processing Pianobook samples from: {self.pianobook_path}")

        pianobook_dir = Path(self.pianobook_path)
        total_organized = {}

        # First, process any Decent Sampler presets to get accurate mappings
        for dspreset in pianobook_dir.rglob('*.dspreset'):
            logger.info(f"Parsing Decent Sampler preset: {dspreset.name}")
            mappings = self.organizer.process_decent_sampler_preset(str(dspreset))

            preset_dir = dspreset.parent
            for rel_path, pitch in mappings:
                sample_path = preset_dir / rel_path
                if sample_path.exists():
                    folder_name = PitchMapper.get_folder_name(pitch)
                    dest_folder = self.output_dir / folder_name
                    dest_folder.mkdir(parents=True, exist_ok=True)

                    content_hash = hashlib.md5(sample_path.read_bytes()).hexdigest()[:8]
                    pack_name = dspreset.stem.replace(' ', '_')[:20]
                    new_filename = f"pianobook_{pack_name}_{pitch}_{content_hash}{sample_path.suffix}"
                    dest_path = dest_folder / new_filename

                    if not dest_path.exists():
                        shutil.copy2(sample_path, dest_path)
                        total_organized[pitch] = total_organized.get(pitch, 0) + 1

        # Process SFZ files
        for sfz in pianobook_dir.rglob('*.sfz'):
            logger.info(f"Parsing SFZ file: {sfz.name}")
            mappings = self.organizer.process_sfz_file(str(sfz))

            sfz_dir = sfz.parent
            for rel_path, pitch in mappings:
                sample_path = sfz_dir / rel_path
                if sample_path.exists():
                    folder_name = PitchMapper.get_folder_name(pitch)
                    dest_folder = self.output_dir / folder_name
                    dest_folder.mkdir(parents=True, exist_ok=True)

                    content_hash = hashlib.md5(sample_path.read_bytes()).hexdigest()[:8]
                    pack_name = sfz.stem.replace(' ', '_')[:20]
                    new_filename = f"pianobook_{pack_name}_{pitch}_{content_hash}{sample_path.suffix}"
                    dest_path = dest_folder / new_filename

                    if not dest_path.exists():
                        shutil.copy2(sample_path, dest_path)
                        total_organized[pitch] = total_organized.get(pitch, 0) + 1

        # Also scan for any loose WAV files with pitch in filename
        fallback_organized = self.organizer.scan_and_organize(self.pianobook_path, "pianobook")

        for pitch, count in fallback_organized.items():
            total_organized[pitch] = total_organized.get(pitch, 0) + count

        return total_organized

    def process_local_iowa(self) -> Dict[str, int]:
        """Process locally downloaded University of Iowa samples"""
        if not self.iowa_path:
            return {}

        logger.info(f"Processing Iowa samples from: {self.iowa_path}")
        return self.organizer.scan_and_organize(self.iowa_path, "iowa")

    def collect_freesound(self) -> Dict[str, int]:
        """Collect samples from Freesound"""
        if not self.freesound.api_key:
            logger.info("Skipping Freesound (no API key provided)")
            return {}

        logger.info("Searching Freesound for piano samples...")

        # Various search queries for piano samples
        queries = [
            "piano note single",
            "piano sample",
            "piano key",
            "grand piano note",
            "upright piano sample",
        ]

        all_samples = []
        for query in queries:
            samples = self.freesound.search_piano_samples(query, max_pages=5)
            all_samples.extend(samples)
            time.sleep(1)  # Rate limiting

        # Remove duplicates
        seen_ids = set()
        unique_samples = []
        for sample in all_samples:
            if sample['id'] not in seen_ids:
                seen_ids.add(sample['id'])
                unique_samples.append(sample)

        logger.info(f"Found {len(unique_samples)} unique Freesound samples")

        organized = {}
        for sample in unique_samples[:100]:  # Limit to first 100
            pitch = PitchMapper.parse_filename_for_pitch(sample.get('name', ''))
            if pitch:
                result = self.freesound.download_sample(sample)
                if result:
                    data, filename = result
                    self.freesound.save_sample(data, pitch, "freesound", filename)
                    organized[pitch] = organized.get(pitch, 0) + 1
                time.sleep(0.5)  # Rate limiting

        return organized

    def print_instructions(self):
        """Print download instructions for all sources"""
        print("\n" + "="*60)
        print("PIANO SAMPLE COLLECTION INSTRUCTIONS")
        print("="*60)
        print(self.nsynth.get_download_instructions())
        print(self.pianobook.get_download_instructions())
        print(self.iowa.get_download_instructions())
        print(self.freesound.get_api_instructions())

    def run(self, show_instructions: bool = True):
        """Run the full collection process"""
        logger.info("Starting Piano Sample Collection...")

        # Setup folders
        self.setup_folders()

        # Track statistics
        stats = {
            'nsynth': {},
            'pianobook': {},
            'iowa': {},
            'freesound': {},
        }

        # Process local sources
        if self.nsynth_path:
            stats['nsynth'] = self.process_local_nsynth()

        if self.pianobook_path:
            stats['pianobook'] = self.process_local_pianobook()

        if self.iowa_path:
            stats['iowa'] = self.process_local_iowa()

        # Process online sources
        if self.freesound.api_key:
            stats['freesound'] = self.collect_freesound()

        # Print summary
        self.print_summary(stats)

        # Show instructions if no local paths provided
        if show_instructions and not any([self.nsynth_path, self.pianobook_path,
                                           self.iowa_path, self.freesound.api_key]):
            self.print_instructions()

        return stats

    def print_summary(self, stats: Dict):
        """Print collection summary"""
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)

        total_by_pitch = {}
        total_by_source = {}

        for source, pitch_counts in stats.items():
            source_total = sum(pitch_counts.values())
            total_by_source[source] = source_total

            for pitch, count in pitch_counts.items():
                total_by_pitch[pitch] = total_by_pitch.get(pitch, 0) + count

        print("\nBy Source:")
        for source, total in total_by_source.items():
            print(f"  {source}: {total} samples")

        print(f"\nTotal Samples: {sum(total_by_source.values())}")
        print(f"Pitches with samples: {len(total_by_pitch)}")

        if total_by_pitch:
            print("\nTop 10 pitches by sample count:")
            sorted_pitches = sorted(total_by_pitch.items(), key=lambda x: x[1], reverse=True)[:10]
            for pitch, count in sorted_pitches:
                print(f"  {pitch}: {count}")

        # Check which pitches are missing
        all_pitches = set(PitchMapper.get_all_pitches())
        covered_pitches = set(total_by_pitch.keys())
        missing = all_pitches - covered_pitches

        if missing:
            print(f"\nPitches without samples: {len(missing)}")
            if len(missing) <= 20:
                print(f"  Missing: {', '.join(sorted(missing))}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download and organize piano samples by pitch (A0-C8)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show download instructions
  python piano_sample_downloader.py --instructions

  # Process local NSynth download
  python piano_sample_downloader.py --nsynth-path /path/to/nsynth-test

  # Process Pianobook samples
  python piano_sample_downloader.py --pianobook-path /path/to/pianobook_packs

  # Use Freesound API
  python piano_sample_downloader.py --freesound-key YOUR_API_KEY

  # Combine multiple sources
  python piano_sample_downloader.py --nsynth-path ./nsynth --pianobook-path ./pianobook --output ./notes
        """
    )

    parser.add_argument('--output', '-o', default='notes',
                        help='Output directory for organized samples (default: notes)')
    parser.add_argument('--nsynth-path',
                        help='Path to extracted NSynth dataset')
    parser.add_argument('--pianobook-path',
                        help='Path to extracted Pianobook sample packs')
    parser.add_argument('--iowa-path',
                        help='Path to University of Iowa samples')
    parser.add_argument('--freesound-key',
                        help='Freesound API key')
    parser.add_argument('--instructions', action='store_true',
                        help='Show download instructions and exit')
    parser.add_argument('--scan-path',
                        help='Scan a directory for audio files and organize by pitch')
    parser.add_argument('--scan-source', default='local',
                        help='Source name for scanned files (default: local)')

    args = parser.parse_args()

    # Create collector
    collector = PianoSampleCollector(
        output_dir=args.output,
        freesound_key=args.freesound_key,
        nsynth_path=args.nsynth_path,
        pianobook_path=args.pianobook_path,
        iowa_path=args.iowa_path,
    )

    if args.instructions:
        collector.print_instructions()
        return

    # Handle simple scan mode
    if args.scan_path:
        collector.setup_folders()
        organized = collector.organizer.scan_and_organize(args.scan_path, args.scan_source)
        print(f"\nOrganized {sum(organized.values())} samples across {len(organized)} pitches")
        return

    # Run full collection
    collector.run()


if __name__ == "__main__":
    main()
