import os
import re
import csv
import wave
import hashlib
from collections import Counter, defaultdict

PROJECT_ROOT = "/home/sahith/projects/minor_project_408"
ASTHMA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "asthma_detection_v2",
    "Asthma Detection Dataset Version 2"
)
ICBHI_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "audio_and_text_files"
)
OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "audit"
)

CLASSES = [
    "asthma",
    "Bronchial",
    "copd",
    "healthy",
    "pneumonia"
]

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_audio_information(path):
    try:
        with wave.open(path, "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            duration = frame_count / sample_rate if sample_rate > 0 else 0.0

            frames = audio.readframes(frame_count)

            audio_hash = hashlib.sha256(
                str(channels).encode()
                + str(sample_width).encode()
                + str(sample_rate).encode()
                + frames
            ).hexdigest()

            return {
                "status": "OK",
                "channels": channels,
                "sample_width": sample_width,
                "sample_rate": sample_rate,
                "frames": frame_count,
                "duration": duration,
                "audio_hash": audio_hash
            }

    except Exception as error:
        return {
            "status": "ERROR",
            "channels": "",
            "sample_width": "",
            "sample_rate": "",
            "frames": "",
            "duration": "",
            "audio_hash": "",
            "error": str(error)
        }


def extract_patient_id(filename):
    match = re.match(r"P(\d+)", filename, re.IGNORECASE)
    if match:
        return "P" + match.group(1)
    return ""


def get_wav_files(root):
    wav_files = []

    if not os.path.exists(root):
        return wav_files

    for current_root, directories, files in os.walk(root):
        for filename in files:
            if filename.lower().endswith(".wav"):
                wav_files.append(os.path.join(current_root, filename))

    return sorted(wav_files)


def print_distribution(records):
    print()
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    counter = Counter(record["class"] for record in records)

    total = sum(counter.values())

    for class_name in CLASSES:
        count = counter.get(class_name, 0)
        percentage = (count / total * 100) if total else 0
        print(f"{class_name:15s}: {count:4d} ({percentage:6.2f}%)")

    print(f"{'Total':15s}: {total:4d}")


def print_patient_distribution(records):
    print()
    print("=" * 70)
    print("PATIENT DISTRIBUTION")
    print("=" * 70)

    class_patients = defaultdict(set)

    for record in records:
        patient_id = record["patient_id"]
        if patient_id:
            class_patients[record["class"]].add(patient_id)

    for class_name in CLASSES:
        patients = sorted(class_patients[class_name])
        print(
            f"{class_name:15s}: "
            f"{len(patients):4d} unique patients"
        )

        if patients:
            print(" " * 17 + ", ".join(patients[:20]))

            if len(patients) > 20:
                print(
                    " " * 17
                    + f"... and {len(patients) - 20} more"
                )


def print_audio_properties(records):
    print()
    print("=" * 70)
    print("AUDIO PROPERTIES")
    print("=" * 70)

    valid_records = [
        record for record in records
        if record["status"] == "OK"
    ]

    sample_rates = Counter(
        record["sample_rate"]
        for record in valid_records
    )

    channels = Counter(
        record["channels"]
        for record in valid_records
    )

    sample_widths = Counter(
        record["sample_width"]
        for record in valid_records
    )

    print()
    print("Sampling rates:")

    for value, count in sorted(sample_rates.items()):
        print(f"  {value} Hz: {count}")

    print()
    print("Channels:")

    for value, count in sorted(channels.items()):
        print(f"  {value}: {count}")

    print()
    print("Sample widths:")

    for value, count in sorted(sample_widths.items()):
        print(f"  {value} bytes: {count}")

    durations = [
        record["duration"]
        for record in valid_records
    ]

    if durations:
        print()
        print("Duration statistics:")
        print(f"  Minimum: {min(durations):.3f} seconds")
        print(f"  Maximum: {max(durations):.3f} seconds")
        print(
            f"  Average: {sum(durations) / len(durations):.3f} seconds"
        )


def print_errors(records):
    errors = [
        record
        for record in records
        if record["status"] != "OK"
    ]

    print()
    print("=" * 70)
    print("CORRUPTED OR UNREADABLE FILES")
    print("=" * 70)

    print(f"Total errors: {len(errors)}")

    for record in errors:
        print()
        print(record["path"])
        print(record.get("error", ""))


def find_duplicates(records):
    raw_hash_groups = defaultdict(list)
    audio_hash_groups = defaultdict(list)

    for record in records:
        if record["status"] != "OK":
            continue

        raw_hash_groups[record["file_hash"]].append(record)
        audio_hash_groups[record["audio_hash"]].append(record)

    raw_duplicates = [
        group
        for group in raw_hash_groups.values()
        if len(group) > 1
    ]

    audio_duplicates = [
        group
        for group in audio_hash_groups.values()
        if len(group) > 1
    ]

    print()
    print("=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)

    print(f"Exact file duplicate groups: {len(raw_duplicates)}")
    print(f"Exact audio duplicate groups: {len(audio_duplicates)}")

    return raw_duplicates, audio_duplicates


def save_records(records):
    output_file = os.path.join(
        OUTPUT_ROOT,
        "asthma_v2_audio_audit.csv"
    )

    fieldnames = [
        "dataset",
        "class",
        "filename",
        "path",
        "patient_id",
        "status",
        "channels",
        "sample_width",
        "sample_rate",
        "frames",
        "duration",
        "file_hash",
        "audio_hash",
        "error"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)

    print()
    print(f"Audit CSV saved to:")
    print(output_file)


def save_duplicates(duplicate_groups):
    output_file = os.path.join(
        OUTPUT_ROOT,
        "asthma_v2_duplicates.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "group_id",
            "filename",
            "class",
            "patient_id",
            "path"
        ])

        for group_id, group in enumerate(
            duplicate_groups,
            start=1
        ):
            for record in group:
                writer.writerow([
                    group_id,
                    record["filename"],
                    record["class"],
                    record["patient_id"],
                    record["path"]
                ])

    print()
    print(f"Duplicate report saved to:")
    print(output_file)


def audit_asthma_dataset():
    print("=" * 70)
    print("ASTHMA DETECTION DATASET VERSION 2 AUDIT")
    print("=" * 70)

    print()
    print(f"Dataset root:")
    print(ASTHMA_ROOT)

    records = []

    for class_name in CLASSES:
        class_root = os.path.join(
            ASTHMA_ROOT,
            class_name
        )

        files = get_wav_files(class_root)

        print()
        print(
            f"Scanning {class_name}: "
            f"{len(files)} WAV files"
        )

        for path in files:
            filename = os.path.basename(path)

            information = get_audio_information(path)

            record = {
                "dataset": "Asthma Detection Dataset Version 2",
                "class": class_name,
                "filename": filename,
                "path": path,
                "patient_id": extract_patient_id(filename),
                "status": information["status"],
                "channels": information["channels"],
                "sample_width": information["sample_width"],
                "sample_rate": information["sample_rate"],
                "frames": information["frames"],
                "duration": information["duration"],
                "file_hash": "",
                "audio_hash": information["audio_hash"],
                "error": information.get("error", "")
            }

            if information["status"] == "OK":
                record["file_hash"] = sha256_file(path)

            records.append(record)

    print_distribution(records)
    print_patient_distribution(records)
    print_audio_properties(records)
    print_errors(records)

    raw_duplicates, audio_duplicates = find_duplicates(records)

    save_records(records)
    save_duplicates(audio_duplicates)

    return records


def compare_with_icbhi(records):
    print()
    print("=" * 70)
    print("ICBHI OVERLAP ANALYSIS")
    print("=" * 70)

    if not os.path.exists(ICBHI_ROOT):
        print()
        print("ICBHI directory was not found:")
        print(ICBHI_ROOT)
        print()
        print("Skipping ICBHI comparison.")
        return

    icbhi_files = get_wav_files(ICBHI_ROOT)

    print()
    print(f"ICBHI WAV files found: {len(icbhi_files)}")

    asthma_file_hashes = set(
        record["file_hash"]
        for record in records
        if record["status"] == "OK"
        and record["file_hash"]
    )

    asthma_audio_hashes = set(
        record["audio_hash"]
        for record in records
        if record["status"] == "OK"
        and record["audio_hash"]
    )

    exact_file_matches = []
    exact_audio_matches = []

    for path in icbhi_files:
        filename = os.path.basename(path)

        try:
            information = get_audio_information(path)

            if information["status"] != "OK":
                continue

            file_hash = sha256_file(path)
            audio_hash = information["audio_hash"]

            if file_hash in asthma_file_hashes:
                exact_file_matches.append(
                    (filename, path)
                )

            if audio_hash in asthma_audio_hashes:
                exact_audio_matches.append(
                    (filename, path)
                )

        except Exception:
            continue

    print()
    print(
        f"Exact file matches between Asthma V2 and ICBHI: "
        f"{len(exact_file_matches)}"
    )

    for filename, path in exact_file_matches[:30]:
        print(f"  {filename}")
        print(f"    {path}")

    print()
    print(
        f"Exact audio matches between Asthma V2 and ICBHI: "
        f"{len(exact_audio_matches)}"
    )

    for filename, path in exact_audio_matches[:30]:
        print(f"  {filename}")
        print(f"    {path}")

    output_file = os.path.join(
        OUTPUT_ROOT,
        "icbhi_overlap.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "match_type",
            "icbhi_filename",
            "icbhi_path"
        ])

        for filename, path in exact_file_matches:
            writer.writerow([
                "exact_file",
                filename,
                path
            ])

        for filename, path in exact_audio_matches:
            writer.writerow([
                "exact_audio",
                filename,
                path
            ])

    print()
    print("ICBHI overlap report saved to:")
    print(output_file)


def main():
    records = audit_asthma_dataset()
    compare_with_icbhi(records)

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print()
    print(f"Reports are available in:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()