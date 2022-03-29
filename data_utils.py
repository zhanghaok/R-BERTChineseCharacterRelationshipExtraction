import re
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))


bert_base_chinese = "/home/zhk/pretrained-model/bert-base-chinese"

class MyTokenizer(object):
    def __init__(self,pretrained_model_path,mask_entity=False):
        self.pretrained_model_path = pretrained_model_path #'/home/zhk/pretrained-model/bert-base-chinese'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.mask_entity = mask_entity

    def tokenize(self,item):
        sentence = item['text']#'-家庭张国立和父亲张默是张默和前妻罗秀春的儿子，据了解，张国立与罗女士相识于少年时代，长'
        pos_head = item['h']['pos']#[17, 20]头实体
        pos_tail = item['t']['pos']#[9, 11]尾巴实体
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail#[9, 11]
            pos_max = pos_head#[17, 20]
            rev = True
        else:
            pos_min = pos_head
            pos_max = pos_tail
            rev = False

        sent0 = self.bert_tokenizer.tokenize(sentence[:pos_min[0]])#['-', '家', '庭', '张', '国', '立', '和', '父', '亲']
        ent0 = self.bert_tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])#['张', '默']
        sent1 = self.bert_tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])#['是', '张', '默', '和', '前', '妻']
        ent1 = self.bert_tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])#['罗', '秀', '春']
        sent2 = self.bert_tokenizer.tokenize(sentence[pos_max[1]:])#['的', '儿', '子', '，', '据', '了', '解', '，', '张', '国', '立', '与', '罗', '女', '士', '相', '识', '于', '少', '年', '时', '代', '，', '长']

        if rev:
            if self.mask_entity:
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_tail = [len(sent0), len(sent0) + len(ent0)]#[9, 11]
            pos_head =[
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]#[17, 20]
        else:
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        #['-', '家', '庭', '张', '国', '立', '和', '父', '亲', '张', '默', '是', '张', '默', '和', '前', '妻', '罗', '秀', '春', '的', '儿', '子', '，', '据', '了', '解', '，', '张', '国', '立', '与', '罗', '女', '士', '相', '识', '于', '少', '年', '时', '代', '，', '长']
        tokens = sent0 + ent0 +sent1 + ent1 +sent2

        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = [0,0]#[20, 25]罗秀春
        pos2 = [0,0]#[10, 14][20, 25]
        for token in tokens:#token第9个是张默的张，第17个token是罗秀春的罗
            token = token.lower()
            if cur_pos == pos_head[0]:#罗满足 17== 17
                pos1[0] = len(re_tokens)
                re_tokens.append('[unused1]')
            if cur_pos == pos_tail[0]:#满足9 == 9
                pos2[0] = len(re_tokens)
                re_tokens.append('[unused2]')
            re_tokens.append(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused3]')
                pos1[1] = len(re_tokens)
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused4]')
                pos2[1] = len(re_tokens)
            cur_pos += 1
        re_tokens.append('[SEP]')
        #['[CLS]', '-', '家', '庭', '张', '国', '立', '和', '父', '亲', '[unused2]', '张', '默', '[unused4]', '是', '张', '默', '和', '前', '妻', '[unused1]', '罗', '秀', '春', '[unused3]', '的', '儿', '子', '，', '据', '了', '解', '，', '张', '国', '立', '与', '罗', '女', '士', '相', '识', '于', '少',,,'[SEP]']
        return re_tokens[1:-1],pos1,pos2



def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask

def read_data(input_file, tokenizer=None, max_len=128):
    """
    读取train_small.jsonl或者val_small.jsonl
    序列最长长度= 128
    """
    tokens_list = []
    e1_mask_list = []
    e2_mask_list = []
    tags = []
    with open(input_file, 'r', encoding='utf-8') as f_in:#'/home/zhk/data/RE/train_small.jsonl'
        for line in tqdm(f_in):
            line = line.strip()#去掉最后的\n
            item = json.loads(line)
            if tokenizer is None:
                tokenizer = MyTokenizer(bert_base_chinese)#上面定义的分词器
            tokens, pos_e1, pos_e2 = tokenizer.tokenize(item) #['-', '家', '庭', '张', '国', '立', '和', '父', '亲', '[unused2]', '张', '默', '[unused4]', '是', '张', '默', '和', '前', '妻', '[unused1]', '罗', '秀', '春', '[unused3]', '的', '儿', '子', '，', '据', '了', '解', '，', '张', '国', '立', '与', '罗', '女', '士', '相', '识', '于', '少', '年', '时','代','长' #pos_e1:[20, 25]#pos_e2:[[10, 14]]
            if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and \
                    pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
                tokens_list.append(tokens)#去除cls和sep
                e1_mask = convert_pos_to_mask(pos_e1, max_len)#([20, 25],128)长度为128的全零list，但是在entity1的位置为1
                e2_mask = convert_pos_to_mask(pos_e2, max_len)
                e1_mask_list.append(e1_mask)
                e2_mask_list.append(e2_mask)
                tag = item['relation']
                tags.append(tag)
    return tokens_list, e1_mask_list, e2_mask_list,tags



def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))

def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))

def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


class SentenceREDataset(Dataset):
    def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=128):
        self.data_file_path = data_file_path
        self.tagset_path = tagset_path
        self.pretrained_model_path = pretrained_model_path
        self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)
        self.max_len = max_len
        self.tokens_list, self.e1_mask_list, self.e2_mask_list, self.tags = read_data(data_file_path, tokenizer=self.tokenizer, max_len=self.max_len)
        self.tag2idx = get_tag2idx(self.tagset_path)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):#idx=0
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        sample_e1_mask = self.e1_mask_list[idx]
        sample_e2_mask = self.e2_mask_list[idx]
        sample_tag = self.tags[idx]#关系的标签：父母
        encoded = self.tokenizer.bert_tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']
        sample_tag_id = self.tag2idx[sample_tag]

        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'e1_mask': torch.tensor(sample_e1_mask),
            'e2_mask': torch.tensor(sample_e2_mask),
            'tag_id': torch.tensor(sample_tag_id)
        }
        return sample


# # print(len("-家庭张国立和父亲张默是张默和前妻罗秀春的儿子，据了解，张国立与罗女士相识于少年时代，长"))
#
# print(len([i for i in token_type_ids if i==1]))
# print(len([i for i in attention_mask if i==1]))
#
# print(1111)