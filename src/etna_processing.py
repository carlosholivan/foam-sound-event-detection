import os
from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path
import csv
from time import sleep
from rich.progress import BarColumn, ProgressColumn, Progress
from rich.console import Console
import pandas as pd
import threading
import matplotlib.pyplot as plt
import shutil


def cut_audios(base_path: Path, dest_path: Path):
    # Create the destination directory if it doesn't exist
    dest_path = Path(dest_path, "audios")
    if not os.path.exists(str(dest_path)):
        os.makedirs(dest_path)

    # Get the file name prefix from each XLSX file in the base path
    noms = []
    for xlsx_file in base_path.glob("**/*.xlsx"):
        filename = xlsx_file.stem.split("_")
        noms.append(filename[0] + "_" + filename[1])

    class CustomColumn(ProgressColumn):
        def render(self, task):
            return f"{task.description} ({task.percentage:.2f}% complete)"

    progress = Progress(
        "{task.description}",
        BarColumn(bar_width=50),
        CustomColumn(),
        "{task.completed}/{task.total}",
        console=Console(),
        auto_refresh=False,
    )

    def process_wav(nom):
        num_chunks = 0
        for wav_file_path in base_path.glob("**/*.wav"):
            # Only process files with the given prefix
            if not wav_file_path.name.startswith(nom):
                continue
            audio = AudioSegment.from_wav(wav_file_path)
            audio_name = Path(wav_file_path).stem
            size = 10000  # Cut the file into chunks of 10 seconds
            chunks = make_chunks(audio, size)
            for i, chunk in enumerate(chunks):
                chunk_name = f"{audio_name}_{i}.wav"
                chunk_path = Path(dest_path) / chunk_name
                chunk.export(chunk_path, format="wav")
                num_chunks += 1

    threads = []
    task = progress.add_task("Processing audio files", total=len(noms))
    for nom in noms:
        t = threading.Thread(target=process_wav, args=(nom,))
        t.start()
        threads.append(t)
    with progress:
        while any(t.is_alive() for t in threads):
            progress.update(task, advance=1)
            sleep(0.1)


def label_chunks(base_path: Path, dest_path: Path):
    filename = "labels.csv"
    dest_labels_path = Path(dest_path, "labels")
    dest_audios_path = Path(dest_path, "audios")
    if not os.path.exists(dest_labels_path):
        os.mkdir(dest_labels_path)
    destination = Path(dest_labels_path, filename)

    label_1 = "No desbordamiento"
    label_2 = "Desbordamiento"

    # Cogemos la nomenclatura que relaciona el csv con su audio
    noms = []
    for csv_file in base_path.glob("*.xlsx"):
        filename = csv_file.stem.split("_")
        noms.append(filename[0] + "_" + filename[1])

    # Recorremos todos los archivos wav del directorio y los cortamos en trozos de 10 segundos
    all_labels = []
    for wav_file_path in base_path.glob("*.wav"):
        for nom in noms:
            if nom not in wav_file_path.name:
                continue

            # Leemos el archivo
            file_path = Path(base_path, nom + ".xlsx")

            def convert_to_10sec_splits(file_path):
                df = pd.read_excel(file_path, "Interaction")
            
                # convert the Timestamp column to a datetime object
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])

                # Set the 'Timestamp' column as the dataframe index
                df.set_index('Timestamp', inplace=True)

                # Resample the dataframe using a 10-second interval and join the rows in each interval
                df_resampled = df.resample('10S')['Step', 'Texto'].agg(lambda x: ', '.join(x))

                # Reset the index of the resampled dataframe and fill any missing values with 'NaN'
                df_resampled.reset_index(inplace=True)
                df_resampled.fillna('NaN', inplace=True)

                # Reset the index of the resampled dataframe and fill any missing values with 'NaN'
                df_resampled.reset_index(inplace=True)
                df_resampled.fillna('NaN', inplace=True)

                # Create a new 'Crono' column with intervals of 10 seconds
                df_resampled['Crono'] = pd.to_timedelta(df_resampled.index * 10, unit='s')

                # Reorder the columns to match the original dataframe
                df_resampled = df_resampled[['Timestamp', 'Crono', 'Step', 'Texto']]
                for index, row in df_resampled.iterrows():
                    # if "Text" colum is nan assign the previous value
                    if row[3] == "":
                        df_resampled.loc[index, "Texto"] = df_resampled.loc[index-1, "Texto"]
                    # if "Step" colum is nan assign the previous value
                    if row[2] == "":
                        df_resampled.loc[index, "Step"] = df_resampled.loc[index-1, "Step"]
                return df_resampled

            def convert_to_foam_labels(df):
                for index, row in df.iterrows():
                    # if "Texto" contains "Foam" replace it for "Desbordamiento"
                    # all the next rows will be also replaced by "Desbordamiento"
                    if "Foam" in row[3] or "Boil over" in row[3] and ", " not in row[3]:
                        df.loc[index:, "Texto"] = label_2
                    elif "Foam" in row[3] and ", " in row[3]:
                        df.loc[index:, "Texto"] = f"{label_1}, {label_2}"
                    else:
                        df.loc[index:, "Texto"] = label_1
                return df

            df = convert_to_10sec_splits(file_path)
            # save the dataframe to a csv file
            df.to_csv(destination, index=False)
            df = convert_to_foam_labels(df)
            # save "Texto" as a list
            labels = df["Texto"].tolist()
            
            for i in range(len(labels)):
                all_labels.append([wav_file_path.stem + f"_{i}", labels[i]])
                print(f"Saved labels for file {i+1}/{len(labels)} to {destination}")

    # write list to csv
    with open(destination, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_labels)


def data_analysis(dest_path: Path):
    filename = Path("labels.csv")
    file_path = Path(dest_path, "labels", filename)

    # Leemos el archivo
    df = pd.read_csv(file_path, sep=",", decimal=".", header=None)

    labels = {}
    for _, row in df.iterrows():
        if row[1] in labels.keys():
            labels[row[1]] += 1
        else: 
            labels.update({row[1]: 1})

    # representa histograma y guardalo en file_path
    hist_name = Path(dest_path, filename.stem + "_analysis.png")
    plt.bar(labels.keys(), labels.values(), color='b')
    plt.savefig(hist_name)


def split_data(origin_path: Path, dest_path: Path):
    # Origin path
    dataset_audios_path = Path(origin_path, "audios")
    dataset_labels_path = Path(origin_path, "labels")
    filename = "labels.csv"
    file_path = Path(dataset_labels_path, filename)
    dataset_audios_path = Path(origin_path, "audios")

    dest_audio_path = Path(dest_path, "audio")
    dest_labels_path = Path(dest_path, "metadata")
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    if not os.path.exists(dest_audio_path):
        os.mkdir(dest_audio_path)
    if not os.path.exists(dest_labels_path):
        os.mkdir(dest_labels_path)

    dest_train_audio_path = Path(dest_audio_path, "train")
    dest_validation_audio_path = Path(dest_audio_path, "validation")
    dest_test_audio_path = Path(dest_audio_path, "eval")
    if not os.path.exists(dest_train_audio_path):
        os.mkdir(dest_train_audio_path)
    if not os.path.exists(dest_validation_audio_path):
        os.mkdir(dest_validation_audio_path)
    if not os.path.exists(dest_test_audio_path):
        os.mkdir(dest_test_audio_path)
        
    dest_train_weak_audio_path = Path(dest_train_audio_path, "weak")
    if not os.path.exists(dest_train_weak_audio_path):
        os.mkdir(dest_train_weak_audio_path)

    dest_train_labels_path = Path(dest_labels_path, "train")
    dest_validation_labels_path = Path(dest_labels_path, "validation")
    dest_test_labels_path = Path(dest_labels_path, "eval")
    if not os.path.exists(dest_train_labels_path):
        os.mkdir(dest_train_labels_path)
    if not os.path.exists(dest_validation_labels_path):
        os.mkdir(dest_validation_labels_path)
    if not os.path.exists(dest_test_labels_path):
        os.mkdir(dest_test_labels_path)

    train_split = .7
    val_split = .1

    df = pd.read_csv(file_path, sep=",", decimal=".", header=None)

    # loop in rows 
    files = {}
    for _, row in df.iterrows():
        if row[1] in files.keys():
            files[row[1]].append(row[0])
        else: 
            files.update({row[1]: [row[0]]})

    files_split = {"train": [], "validation": [], "eval": []}
    for key in files.keys():
        training = files[key][:int(len(files[key])*train_split)] #[1, 2, 3, 4, 5, 6, 7, 8]
        validation = files[key][int(len(files[key])*train_split):int(len(files[key])*(train_split+val_split))] #[10]
        testing = files[key][int(len(files[key])*(train_split+val_split)):] #[10]
        files_split["train"].extend(training)
        files_split["validation"].extend(validation)
        files_split["eval"].extend(testing)

    # move audio files
    for i, split in enumerate(files_split.keys()):
        for f in files_split[split]:
            orig_file = Path(dataset_audios_path, f + ".wav")
            if split == "train":
                dest_file = Path(dest_audio_path, split, "weak")
                if not os.path.exists(dest_file):
                    os.mkdir(dest_file)
            else:
                dest_file = Path(dest_audio_path, split)
            # look for label of file
            for k, v in files.items():
                for val in v:
                    if val == f:
                        label = k
            try:
                destination = Path(dest_labels_path, split, split + ".csv")
                with open(destination, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f, label])
                    print(f"Wrote file {destination}")
                shutil.move(orig_file, dest_file)
                print(f"Moved audio file from {orig_file} to {dest_file}")
            except:
                continue


if __name__ == "__main__":
    # Pipeline:
    # 1. cut_audios cuts the entire recording into slices of 10 seconds
    # 2 - label_chunks: Creates a csv file with the chunkname and labels in "trozos/labels" directory
    # 3 - data_analysis: Analyzes the number of files per label of csv, saves histogram in "trozos/labels"
    # 4 - split_data: splits the data in train (70), validation (10) and test (20) sets
    import time
    start_time = time.time()
    dataset_path = Path("F:/bsh/etna_database")
    base_path = Path(dataset_path, "00_NEW DB")
    dest_path = Path(dataset_path, "trozos")
    split_dataset = Path(dataset_path, "data")
    # TODO: bucle en todos los directorios de la base de datos
    cut_audios(base_path, dest_path)
    label_chunks(base_path, dest_path)
    data_analysis(dest_path)
    split_data(dest_path, split_dataset)
    data_analysis(dest_path)
    print(f"Dataset processed in {time.time() - start_time} seconds.")