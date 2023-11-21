vocab_path = 'new_dict.txt'
new_mapper_path = 'new_mapper.txt'
origin_sentence_path = 'sentence.txt'
word_dict = {}

if __name__ == '__main__':
    with open(vocab_path, 'r', encoding='utf-8') as f:
        words_id = f.readline()
        while words_id:
            word, id = words_id.strip().split(' ')
            word_dict[word] = int(id)
            words_id = f.readline()

    with open(new_mapper_path, 'w', encoding='utf-8') as writer:
        with open(origin_sentence_path, 'r', encoding='utf-8') as reader:
            sentence = reader.readline()
            while sentence:
                words = sentence.strip().split(',')
                ids = [word_dict[word] for word in words]
                writer.write(','.join(words))
                writer.write(','.join([str(id) for id in ids]))

                sentence = reader.readline()
