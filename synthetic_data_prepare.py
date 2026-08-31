# ============================================================
# Install dependencies
# ============================================================

!pip install -q soundfile pandas datasets huggingface_hub


# ============================================================
# Imports
# ============================================================

import os
import csv
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download


# ============================================================
# Transcript normalization
# ============================================================

def normalize_transcript(text):
    """
    Normalize transcript text for SpeechBrain CSV loading.

    SpeechBrain interprets expressions such as:

        $10.
        $20
        $5.50

    as replacement variables when loading CSV files.

    Therefore, currency symbols are converted to words.

    Example:

        "The toy costs $10."

    becomes:

        "The toy costs 10 dollars."
    """

    text = str(text)

    # --------------------------------------------------------
    # Convert dollar amounts
    # --------------------------------------------------------

    # $10.50 -> 10 dollars and 50 cents
    text = re.sub(
        r"\$(\d+)\.(\d{2})",
        r"\1 dollars and \2 cents",
        text,
    )

    # $10. -> 10 dollars
    text = re.sub(
        r"\$(\d+)\.",
        r"\1 dollars.",
        text,
    )

    # $10 -> 10 dollars
    text = re.sub(
        r"\$(\d+)",
        r"\1 dollars",
        text,
    )

    # --------------------------------------------------------
    # Convert remaining dollar symbols
    # --------------------------------------------------------

    text = text.replace(
        "$",
        "dollars "
    )

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# Prepare SpeechBrain dataset
# ============================================================

def prepare_speechbrain_dataset(
    repo_id="imesh7/asr_dataset_syn",
    local_dir="/root/hf_datasets/asr_dataset_syn",
    output_dir="/root/asr_dataset",
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=7775,
):
    """
    Download a Hugging Face ASR dataset and convert it into
    SpeechBrain-compatible CSV files.

    Source dataset expected columns:

        file_name
        text
        speaker
        duration
        sample_rate
        words

    Output:

        output_dir/
            train.csv
            valid.csv
            test.csv
    """

    # ========================================================
    # Validate split ratios
    # ========================================================

    if abs(
        train_ratio
        + valid_ratio
        + test_ratio
        - 1.0
    ) > 1e-6:

        raise ValueError(
            "train_ratio + valid_ratio + test_ratio "
            "must equal 1.0"
        )

    # ========================================================
    # Convert directories to Path objects
    # ========================================================

    local_dir = Path(
        local_dir
    )

    output_dir = Path(
        output_dir
    )

    # ========================================================
    # Create directories
    # ========================================================

    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Download Hugging Face dataset
    # ========================================================

    print("=" * 70)
    print("Downloading Hugging Face dataset")
    print("=" * 70)

    print(
        f"Repository : {repo_id}"
    )

    print(
        f"Local dir  : {local_dir}"
    )

    dataset_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )

    dataset_path = Path(
        dataset_path
    )

    print(
        f"Dataset path: {dataset_path}"
    )

    # ========================================================
    # Find metadata CSV
    # ========================================================

    csv_files = list(
        dataset_path.rglob("*.csv")
    )

    if not csv_files:

        raise FileNotFoundError(
            f"No CSV file found inside "
            f"{dataset_path}"
        )

    print("\nCSV files found:")

    for csv_file in csv_files:

        print(
            f"  {csv_file}"
        )

    # Use the first CSV as metadata
    metadata_file = csv_files[0]

    print(
        f"\nUsing metadata file:"
        f"\n{metadata_file}"
    )

    # ========================================================
    # Read metadata CSV
    # ========================================================

    df = pd.read_csv(
        metadata_file
    )

    print("\nDataset columns:")

    print(
        df.columns.tolist()
    )

    print(
        f"\nTotal samples: {len(df)}"
    )

    # ========================================================
    # Validate required columns
    # ========================================================

    required_columns = [
        "file_name",
        "text",
        "speaker",
        "duration",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # ========================================================
    # Remove rows with missing text
    # ========================================================

    missing_text = (
        df["text"]
        .isna()
        .sum()
    )

    if missing_text > 0:

        print(
            f"\nWARNING: {missing_text} samples "
            "have missing text."
        )

        df = df[
            df["text"].notna()
        ].copy()

    # ========================================================
    # Resolve audio paths
    # ========================================================

    print("\n" + "=" * 70)
    print("Resolving audio files")
    print("=" * 70)

    def resolve_audio_path(file_name):

        file_name = str(
            file_name
        )

        # ----------------------------------------------------
        # Dataset contains paths such as:
        #
        # /wav/sample_0000.wav
        #
        # Convert to:
        #
        # wav/sample_0000.wav
        # ----------------------------------------------------

        relative_path = (
            file_name.lstrip("/")
        )

        # ----------------------------------------------------
        # First try expected path
        # ----------------------------------------------------

        candidate = (
            dataset_path
            / relative_path
        )

        if candidate.exists():

            return candidate

        # ----------------------------------------------------
        # Fallback: search by filename
        # ----------------------------------------------------

        filename = Path(
            relative_path
        ).name

        matches = list(
            dataset_path.rglob(
                filename
            )
        )

        if matches:

            return matches[0]

        return None

    print(
        "Searching for audio files..."
    )

    df["audio_path"] = [
        resolve_audio_path(
            file_name
        )
        for file_name in df["file_name"]
    ]

    # ========================================================
    # Check missing audio
    # ========================================================

    missing_audio = df[
        df["audio_path"].isna()
    ]

    if len(missing_audio) > 0:

        print(
            f"\nWARNING: {len(missing_audio)} "
            "audio files could not be found."
        )

        print(
            "\nFirst missing files:"
        )

        for file_name in missing_audio[
            "file_name"
        ].head(10):

            print(
                f"  {file_name}"
            )

        # Remove missing audio
        df = df[
            df["audio_path"].notna()
        ].copy()

    print(
        f"\nUsable samples: {len(df)}"
    )

    if len(df) == 0:

        raise RuntimeError(
            "No valid audio files were found."
        )

    # ========================================================
    # Shuffle dataset
    # ========================================================

    df = df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(
        drop=True
    )

    # ========================================================
    # Train / validation / test split
    # ========================================================

    n = len(df)

    train_end = int(
        n * train_ratio
    )

    valid_end = (
        train_end
        + int(n * valid_ratio)
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    valid_df = df.iloc[
        train_end:valid_end
    ].copy()

    test_df = df.iloc[
        valid_end:
    ].copy()

    print("\n" + "=" * 70)
    print("Dataset split")
    print("=" * 70)

    print(
        f"Train      : {len(train_df)}"
    )

    print(
        f"Validation : {len(valid_df)}"
    )

    print(
        f"Test       : {len(test_df)}"
    )

    # ========================================================
    # Write SpeechBrain CSV
    # ========================================================

    def write_speechbrain_csv(
        dataframe,
        output_csv,
    ):

        output_csv = Path(
            output_csv
        )

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(
                f,
                quoting=csv.QUOTE_MINIMAL,
            )

            # ------------------------------------------------
            # SpeechBrain header
            # ------------------------------------------------

            writer.writerow([
                "ID",
                "duration",
                "wav",
                "spk_id",
                "wrd",
            ])

            # ------------------------------------------------
            # Write samples
            # ------------------------------------------------

            for idx, row in dataframe.iterrows():

                audio_path = Path(
                    row["audio_path"]
                )

                # Normalize transcript
                text = normalize_transcript(
                    row["text"]
                )

                writer.writerow([
                    f"sample_{idx:06d}",

                    float(
                        row["duration"]
                    ),

                    str(
                        audio_path.resolve()
                    ),

                    str(
                        row["speaker"]
                    ),

                    text,
                ])

        print(
            f"Created: {output_csv}"
        )

    # ========================================================
    # Generate CSV files
    # ========================================================

    train_csv = (
        output_dir
        / "train.csv"
    )

    valid_csv = (
        output_dir
        / "valid.csv"
    )

    test_csv = (
        output_dir
        / "test.csv"
    )

    write_speechbrain_csv(
        train_df,
        train_csv,
    )

    write_speechbrain_csv(
        valid_df,
        valid_csv,
    )

    write_speechbrain_csv(
        test_df,
        test_csv,
    )

    # ========================================================
    # Validate generated CSV files with pandas
    # ========================================================

    print("\n" + "=" * 70)
    print("Validating generated CSV files")
    print("=" * 70)

    for csv_path in [
        train_csv,
        valid_csv,
        test_csv,
    ]:

        check_df = pd.read_csv(
            csv_path
        )

        expected_columns = [
            "ID",
            "duration",
            "wav",
            "spk_id",
            "wrd",
        ]

        if (
            check_df.columns.tolist()
            != expected_columns
        ):

            raise ValueError(
                f"\nInvalid CSV format: "
                f"{csv_path}\n"
                f"Expected: "
                f"{expected_columns}\n"
                f"Got: "
                f"{check_df.columns.tolist()}"
            )

        print(
            f"\n{csv_path.name}"
        )

        print(
            f"  Samples : {len(check_df)}"
        )

        print(
            f"  Columns : "
            f"{check_df.columns.tolist()}"
        )

    # ========================================================
    # Check for remaining dollar symbols
    # ========================================================

    print("\n" + "=" * 70)
    print("Checking transcript normalization")
    print("=" * 70)

    for csv_path in [
        train_csv,
        valid_csv,
        test_csv,
    ]:

        check_df = pd.read_csv(
            csv_path
        )

        dollar_count = (
            check_df["wrd"]
            .str.contains(
                r"\$",
                regex=True,
                na=False,
            )
            .sum()
        )

        print(
            f"{csv_path.name}: "
            f"{dollar_count} remaining '$' symbols"
        )

    # ========================================================
    # Show example
    # ========================================================

    print("\n" + "=" * 70)
    print("Example training sample")
    print("=" * 70)

    example_df = pd.read_csv(
        train_csv
    )

    print(
        example_df.iloc[0].to_dict()
    )

    # ========================================================
    # Return information
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "SpeechBrain dataset preparation completed."
    )
    print("=" * 70)

    return {
        "dataset_dir": str(
            dataset_path
        ),

        "output_dir": str(
            output_dir
        ),

        "train_csv": str(
            train_csv
        ),

        "valid_csv": str(
            valid_csv
        ),

        "test_csv": str(
            test_csv
        ),

        "num_train": len(
            train_df
        ),

        "num_valid": len(
            valid_df
        ),

        "num_test": len(
            test_df
        ),
    }
