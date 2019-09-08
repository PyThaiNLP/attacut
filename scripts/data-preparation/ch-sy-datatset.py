import fire
import numpy as np

from attacut import preprocessing, utils

DATA_PATH = "../slim-cut/data/best"
SYALLABLE_TOKENIZED_DATA = "./data/best-syllabled-tokenized"

CHARACTER_DICT = "./attacut/artifacts/attacut-sc/characters.json"
SYLLABLE_DICT = "./attacut/artifacts/attacut-sc/syllables.json"


def get_actual_filename(path: str) -> str:
    return "%s/%s" % (SYALLABLE_TOKENIZED_DATA, path.split("/")[-1])


def prepare_syllable_charater_seq_data(files, ch2ix, sy2ix, sampling=10, output_dir=""):
    training, validation = files

    if sampling:
        training = training[:sampling]
        validation = validation[:sampling]

    output_dir = "%s/best-syllable-crf-and-character-seq-feature-sampling-%d" % (output_dir, sampling)

    print("Saving data to %s" % output_dir)
    utils.maybe_create_dir(output_dir)
    for name, dataset in zip(("training", "val"), (training, validation)):
        print("working on : %s" % name)
        fout_txt = open("%s/%s.txt" % (output_dir, name), "w")
        try:
            for path in dataset:
                count = 0
                with open(path, "r") as fin, open(path.replace(".txt", ".label"), "r") as flab:

                    has_space_problem = False

                    for txt, label in zip(fin, flab):
                        txt = txt.strip().replace("~~", "~")

                        if not txt:
                            continue

                        label = label.strip()
                        syllables = txt.split("~")

                        chars_idx = []
                        char_labels = []
                        syllable_idx = []

                        syllable_indices = list(map(
                            lambda sy: preprocessing.syllable2ix(sy2ix, sy),
                            syllables
                        ))

                        if len(syllables) != len(label):
                            print(txt, path)
                            print(len(syllables), len(label))
                            print(syllables)
                            print(label)
                            raise SystemExit("xx")

                        label = list(label)
                        for ii, (syllable, six, l) in enumerate(zip(syllables, syllable_indices, label)):
                            if not syllable:
                                continue

                            if syllable == " ":
                                # next syllable is B, then we should also split this space
                                if label[ii+1] == "1":
                                    l = "1"
                                else:
                                    l = "0"

                            chs = list(
                                map(
                                    lambda c: preprocessing.character2ix(ch2ix, c),
                                    list(syllable)
                                )
                            )

                            total_chs = len(chs)
                            syllable_idx.extend([six] * total_chs)

                            chars_idx.extend(chs)
                            if l == "1":
                                char_labels.extend(["1"] + ["0"] * (total_chs-1))
                            else:
                                char_labels.extend(["0"] * total_chs)

                        assert len(char_labels) == len(chars_idx)

                        # check space problem
                        if not has_space_problem:
                            for cix, clb in zip(chars_idx, char_labels):
                                if cix == 3 and clb == "0":
                                    has_space_problem = True
                                    print(txt)
                                    break

                        fout_txt.write("%s::%s::%s\n" % (
                            "".join(char_labels), 
                            " ".join(np.array(chars_idx).astype(str)),
                            " ".join(np.array(syllable_idx).astype(str)),
                        ))

                    if has_space_problem:
                        print("problem with space in %s" % path)

        finally:
            fout_txt.close()


def main(sampling=10, output_dir="./data"):
    with open("%s/training.files" % DATA_PATH, "r") as f:
        training_files = []
        for l in f:
            training_files.append(get_actual_filename(l.strip()))

    with open("%s/validation.files" % DATA_PATH, "r") as f:
        val_files = []
        for l in f:
            val_files.append(get_actual_filename(l.strip()))

    ch2ix = utils.load_dict(CHARACTER_DICT)
    sy2ix = utils.load_dict(SYLLABLE_DICT)

    prepare_syllable_charater_seq_data(
        (training_files, val_files),
        ch2ix,
        sy2ix,
        sampling=sampling,
        output_dir=output_dir
    )


if __name__ == "__main__":
    fire.Fire(main)
