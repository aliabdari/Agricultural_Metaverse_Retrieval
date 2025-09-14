import json
import os

import inflect
import pickle


def get_rank_exp(num):
    if num == 1:
        return 'first'
    elif num == 2:
        return 'second'
    elif num == 3:
        return 'third'
    elif num == 4:
        return 'fourth'
    elif num == 5:
        return 'fifth'
    elif num == 6:
        return 'sixth'
    elif num == 7:
        return 'seventh'
    else:
        return 'last'


def num_to_words(num):
    p = inflect.engine()
    return p.number_to_words(num)


def process_museums(mus):
    descs = list()
    for m in mus:
        desc_tmp = 'In this museum there are ' + num_to_words(len(m['rooms'])) + ' rooms. '
        for idx_r, r in enumerate(m['rooms']):
            desc_tmp += 'In the ' + get_rank_exp(idx_r + 1) + ' room there are ' + num_to_words(
                len(m['rooms'][r])) + ' videos about ' + m['context'] + ' ' + r + '.'
        descs.append(desc_tmp)
    return descs


if __name__ == '__main__':
    museums = json.load(open('final_museums_ircdl.json', 'r'))

    descriptions = process_museums(museums)
    path_descs = '../descriptions_ircdl'
    os.makedirs(path_descs, exist_ok=True)
    for idx, des in enumerate(descriptions):
        with open(f'{path_descs}/Museum_{str(idx)}.txt', 'w') as file:
            file.write(des)
        pass
