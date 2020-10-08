import xml.etree.ElementTree
import pickle as pkl

UNK = "$UNK$"
NUM = "$NUM$"


class PICODataset_xml(object):
    """
    Class that iterates over CoNLL Dataset

    __iter__ method yields a list [sentences, tags] for each abstract file.
        sentences: list of sentences (untokenized / raw)
        tags: list of integers with P, IC, O, PICO binary annotations

    If processing_sent_words is not None then extra preprocessing is applied.

    Note: this class does not handle splitting the dataset in
    training/validation/testing splits. This operation has to take place
    beforehand.
    """
    def __init__(self,directory, files, processing_sent_words=None, max_iter=None):

        self.directory = directory
        self.files = files
        self.processing_sent_words = processing_sent_words
        self.max_iter = max_iter

    def __iter__(self):
        iter = 0
        for file in self.files:
            sentences, ners, tags = [], [], []
            et = xml.etree.ElementTree.parse(file).getroot().findall('Abstract')
            if len(et) == 0: #no abstract - no iteration counts
                continue #to go  to next loop operation / next file
            elif len(et) > 0 and len(et) < 2 : #just 1 abstract in document(it should be this way)
                iter += 1
                if self.max_iter is not None and iter > self.max_iter:
                    break
                abstract_sentence_objects = et[0].findall('Sentence')
                for sentence_obj in abstract_sentence_objects:
                    sentences.append(sentence_obj.text)
                    ners.append([int(sentence_obj.get('p')), int(sentence_obj.get('ic')),
                                 int(sentence_obj.get('o'))])
                    tags.append([int(sentence_obj.get('pico'))])
                    if self.processing_sent_words is not None:
                        sentences = self.processing_sent_words(sentences)
                yield [sentences, ners, tags]
            elif len(et) > 2:
                for abstract_objects in et:
                    iter +=1
                    if self.max_iter is not None and iter > self.max_iter:
                        break
                    abstract_sentence_objects = abstract_objects.findall('Sentence')

                    for sentence_obj in abstract_sentence_objects:
                        sentences.append(sentence_obj.text)
                        ners.append([int(sentence_obj.get('p')), int(sentence_obj.get('ic')),
                                     int(sentence_obj.get('o'))])
                        tags.append([int(sentence_obj.get('pico'))])
                        if self.processing_sent_words is not None:
                            sentences = self.processing_sent_words(sentences)
                    yield [sentences, ners, tags]

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

class PICODataset_pkl(object):
    """
    Class that iterates over CoNLL Dataset

    __iter__ method yields a list [sentences, tags] for each abstract file.
        sentences: list of sentences (untokenized / raw)
        tags: list of integers with P, IC, O, PICO binary annotations

    If processing_sent_words is not None then extra preprocessing is applied.

    Note: this class does not handle splitting the dataset in
    training/validation/testing splits. This operation has to take place
    beforehand.
    """
    def __init__(self,directory, files, processing_sent_words=None, max_iter=None):

        self.directory = directory
        self.files = files
        self.processing_sent_words = processing_sent_words
        self.max_iter = max_iter

    def __iter__(self):
        for file in self.files:
            with open(file , 'rb') as f:
                data = pkl.load(f)
            f.close()
            yield data

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_processing_sent_words(vocab_words=None, vocab_chars=None,
                              lowercase=False, chars=False, allow_unk=True,
                              transform_nums = True):

    def f(sentences):
        new_sentences = []
        for sentence in sentences:
            words = sentence.split(' ') 
            if lowercase:
                words = [word.lower() for word in words]
            if transform_nums:
                words = [NUM if word.isdigit() else word for word in words]

            if vocab_words is not None and allow_unk:
                words = [vocab_words[word] if word in vocab_words else
                         vocab_words[UNK] for word in words]
            elif vocab_words is not None and not allow_unk:
                for word in words:
                    if word not in vocab_chars:
                        raise Exception('Unknown word key.')
                    #skipping sentence.
            #character will take place after the word transformations.
            if chars and vocab_chars is not None:
                #assume that no unknown chars exist
                char_ids = []
                for word in words:
                    char_ids += [vocab_chars[char] for char in word]

            if chars:
                new_sentences += [char_ids, words]
            else:
                new_sentences += [words]

            return new_sentences

        return  f

def build_pred_with_tags(config):
    import glob

    files = glob.glob(config.file_data_gold+'*.xml')
    print('Building PKL objects from predicted PICO entity labels...')
    for file in files:
        filename = file.split('.')[0].split('/')[-1]
        data_gold = PICODataset_xml(config.file_data_gold, [file], config.processing_sent_words,config.max_iter)
        data_gold = [entry for entry in data_gold][0]
        pred_file = config.file_data_pred+filename+'.txt'
        sentences, pos, tags = [], [], []
        with open(pred_file) as f:
            sentence, sent_tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    sentences.append(sentence)
                    tags.append(sent_tags)
                    sentence, sent_tags = [], []
                else:
                    ls = line.split(' ')
                    word, pos, tag = ls[0], ls[1], ls[-1]
                    sentence.append(word)
                    sent_tags.append(tag)

        new_tags = []
        for tag in tags:
            new_tag = [0, 0, 0]
            if '1_p' in tag:
                new_tag[0] = 1
            if '1_i' in tag:
                new_tag[1] = 1
            if '1_o' in tag:
                new_tag[2] = 1
            new_tags.append(new_tag)

        data_gold[1] == new_tags

        outfile = config.file_data_pred_tags + filename + '.pkl'
        with open(outfile, 'wb') as f:
            pkl.dump(data_gold, f, pkl.HIGHEST_PROTOCOL)
        f.close()

    print('PKL Build completed.')



def save_predicted(save_path,original,predictions,idx2tag):
    doc_words = []
    doc_labels = []

    for i in range(0,len(original)):
        sent_words = []
        sent_labels = []

        contents = original[i][0]
        labels = original[i][1]
        pred_labels = predictions[i]

        for j in range(0,len(labels)):
            if type(labels[j]) != int:
                sent_words.append(contents[j][-1])
                sent_labels.append(idx2tag[pred_labels[j]])
            else:
                doc_words.append(sent_words)
                doc_labels.append(sent_labels)
                break

    with open(save_path, "w+") as f:
        for i in range(0,len(doc_words)): #sents
            for j in range(0,len(doc_words[i])): #entry
                f.write(str(doc_words[i][j])+' '+ 'POS'+ ' '+str(doc_labels[i][j])+ ' \n')
            f.write('\n')
        f.close()

    return True