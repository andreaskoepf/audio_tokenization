import tarfile
from pathlib import Path
import torchaudio



if __name__ == '__main__':
    p = Path("/home/koepf/Downloads/audio1/000001.tar")

    with tarfile.open(p, mode="r") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.lower().endswith(".mp3"):
                print(f"processing {member.name}")

                file_obj = tar.extractfile(member)
                try:
                    if file_obj:
                        x, sr = torchaudio.load(file_obj, format="mp3")
                        print(x.shape, sr)
                except Exception as ex:
                    print(ex)
