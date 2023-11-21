import os
import re

folder_path = 'P_skeleton'
vocab_path = 'new_dict.txt'
sentence_mapper = 'mapz.txt'
new_sentence_mapper = 'sentence_mapper.txt'

word_list = ['<pad>', '<unk>', '<s>', '</s>']
origin_id_dict = {}

if __name__ == '__main__':
    file_names = os.listdir(folder_path)
    pattern_i_w = re.compile(r'(\d+)([\u4e00-\u9fa5]+)')
    pattern_w = re.compile(r'[\u4e00-\u9fa5]+')

    for file_name in file_names:
        pure_name, extension = os.path.splitext(file_name)
        matches = pattern_i_w.match(pure_name)
        if bool(matches):
            id = int(matches.group(1))
            word = matches.group(2)
            origin_id_dict[id] = word
        else:
            word = pattern_w.match(pure_name).group()
        os.rename(os.path.join(os.getcwd(), folder_path, file_name),
                  os.path.join(os.getcwd(), folder_path, f'{len(word_list)}{extension}'))
        word_list.append(word)

    new_word_dict = {word: id for id, word in enumerate(word_list)}

    with open(new_sentence_mapper, 'w', encoding='utf-8') as writer:
        with open(sentence_mapper, 'r', encoding='utf-8') as reader:
            sentence = reader.readline()
            while sentence:
                sentence = sentence.strip().split(',')
                ids = reader.readline().strip().split(',')
                new_ids = []

                try:
                    for id in ids:
                        w = origin_id_dict[int(id)]
                        new_ids.append(new_word_dict[w])

                    assert len(new_ids) == len(sentence)
                    sentence = ','.join(sentence)
                    ids = ','.join([str(id) for id in new_ids])
                    writer.write(f'{sentence}\n')
                    writer.write(f'{ids}\n')
                except:
                    pass
                finally:
                    sentence = reader.readline()

    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word, id in new_word_dict.items():
            f.write(f'{word} {id}\n')

    print(f'create the vocab file, the size of the vocab is {len(word_list)}')
