from random import choice, randrange
import csv


def sample(file_name, file_revert, numberOfStrings, min_length=3, max_length=15):
    with open(file_name, 'w') as f,\
            open(file_revert, 'w') as f_r:
        for i in range(numberOfStrings):
            random_length = randrange(min_length, max_length)  # Pick a random length
            random_char_list = [choice(characters[:-1]) for _ in range(random_length)]  # Pick random chars
            random_string = ' '.join(random_char_list)
            random_revert = ''.join([x for x in random_string[::-1]])
            f.write(random_string + '\n')
            f_r.write(random_revert + '\n')
            print(random_string)
            print(random_revert)


def sample_to_csv(file_name, numberOfStrings, min_length=3, max_length=25):
    fields = ['src', 'trg']
    dict_writer = csv.DictWriter(open(file_name, 'w'), fieldnames=fields)
    dict_writer.writeheader()
    data = {}

    for i in range(numberOfStrings):
        random_length = randrange(min_length, max_length)  # Pick a random length
        random_char_list = [choice(characters[:-1]) for _ in range(random_length)]  # Pick random chars
        random_string = ' '.join(random_char_list)
        random_revert = ''.join([x for x in random_string[::-1]])

        data['src'] = random_string
        data['trg'] = random_revert
        dict_writer.writerow(data)


# generate data for the revert toy
if __name__ == '__main__':
    characters = list("abcd")
    # sample('toy_revert/src-train.txt', 'toy_revert/trg-train.txt', 10000)
    # sample('toy_revert/src-val.txt', 'toy_revert/trg-val.txt', 1000)
    # sample('toy_revert/src-test.txt', 'toy_revert/trg-test.txt', 1000)

    sample_to_csv('toy_revert/train.csv', 20000)
    sample_to_csv('toy_revert/val.csv', 2000)
    sample_to_csv('toy_revert/test.csv', 2000)
