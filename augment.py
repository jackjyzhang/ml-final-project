from datasets import load_from_disk, Dataset
import nlpaug.augmenter.word as naw
from tqdm import tqdm

SEED = 42
OUTPATH = 'data/hate_speech-3splits_aug6-0-1'

# orig train set Counter({1: 15293, 2: 3355, 0: 1178})
# AUG_NUM = {
#     0: 12,
#     1: 0,
#     2: 3
# }
AUG_NUM = {
    0: 6,
    1: 0,
    2: 1
}

def add_example(dict, tweet, class_label):
    dict['tweet'].append(tweet)
    dict['class'].append(class_label)

if __name__ == '__main__':
    aug = naw.ContextualWordEmbsAug(model_path='roberta-large', action="insert", top_k=500, device='cuda')
    ds = load_from_disk('data/hate_speech-3splits')
    orig_train = ds['train']

    augment_dict = {'tweet':[], 'class':[]}
    for example in tqdm(orig_train):
        orig_text, label = example['tweet'], example['class']
        add_example(augment_dict, orig_text, label)

        aug_num = AUG_NUM[label]
        if aug_num > 0:
            aug_texts = aug.augment(orig_text, n=aug_num)
            if aug_num == 1:
                add_example(augment_dict, aug_texts, label)
            else:
                for aug_text in aug_texts:
                    add_example(augment_dict, aug_text, label)
        
    aug_train = Dataset.from_dict(augment_dict)
    aug_train.shuffle(seed=SEED)
    
    print('Examples:')
    for i in range(5):
        print(aug_train[i])
        print('-'*50)

    ds['train'] = aug_train
    ds.save_to_disk(OUTPATH)
