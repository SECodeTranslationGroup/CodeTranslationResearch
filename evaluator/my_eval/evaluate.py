import logging
import sys
import argparse
import Levenshtein
from bleu import _bleu, _bleu_single_sentence


def main():
    parser = argparse.ArgumentParser(description='Evaluate code translation results.')
    parser.add_argument('--sources', '-src', help="filename of the source code.")
    parser.add_argument('--references', '-ref', help="filename of the gold translation.")
    parser.add_argument('--predictions', '-pre', help="filename of the model translation.")
    parser.add_argument('--statistics_file', '-stat', help="filename of the total statistics.")
    parser.add_argument('--output_file', '-out', help="filename of the output.")
    parser.add_argument('--name', '-n', help="name of this model.")

    args = parser.parse_args()

    srcs = []
    refs = []
    pres = []
    output = []
    statistics = []
    with open(args.sources, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            srcs.append(line.strip())
    with open(args.references, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            refs.append(line.strip())
    with open(args.predictions, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            pres.append(line.strip())

    assert len(srcs) == len(refs) and len(refs) == len(pres)

    length = len(refs)

    exact_match_count = 0
    total_bleu = 0
    total_es = 0
    bleu_95_count = 0
    bleu_90_count = 0
    bleu_80_count = 0
    bleu_60_count = 0
    for i in range(length):
        s = srcs[i]
        r = refs[i]
        p = pres[i]
        bleu_score = _bleu_single_sentence(r, p)*100
        edit_sim_score = Levenshtein.distance(r, p)
        total_bleu += bleu_score
        total_es += edit_sim_score
        if r == p:
            exact_match_count += 1
        if bleu_score > 95:
            bleu_95_count += 1
        if bleu_score > 90:
            bleu_90_count += 1
        if bleu_score > 80:
            bleu_80_count += 1
        if bleu_score > 60:
            bleu_60_count += 1
        output.append("Src: ")
        output.append(s)
        output.append("Ref: ")
        output.append(r)
        output.append("Hyp: ")
        output.append(p)
        output.append("")
        output.append("BLEU: " + str(round(bleu_score, 2)))
        output.append("Edit Sim: " + str(edit_sim_score))
        output.append("")
        output.append("")

    statistics.append(args.name)
    statistics.append("Exact Match: " + str(round(exact_match_count / length*100, 2)))
    statistics.append("Avg. BLEU Score: " + str(round(total_bleu / length, 2)))
    statistics.append("Avg. Edit Sim Score: " + str(round(total_es / length, 2)))
    statistics.append("BLEU > 95: " + str(round(bleu_95_count / length * 100, 2)))
    statistics.append("BLEU > 90: " + str(round(bleu_90_count / length * 100, 2)))
    statistics.append("BLEU > 80: " + str(round(bleu_80_count / length * 100, 2)))
    statistics.append("BLEU > 60: " + str(round(bleu_60_count / length * 100, 2)))
    statistics.append("")

    with open(args.statistics_file, 'a', encoding='utf-8') as f:
        for lines in statistics:
            f.write(lines + '\n')

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for lines in output:
            f.write(lines + '\n')


if __name__ == '__main__':
    main()
