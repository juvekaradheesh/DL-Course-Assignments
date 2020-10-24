from PIL import Image
import nltk
import numpy as np
import torch
import torch.utils.data as data
import os, json
# nltk.download('punkt')

class Flickr30k(data.Dataset):
    """Flickr30k dataset"""
    
    def __init__(self, split, vocab=None, root=None, transform=None):
        """
        Args:
            vocab: vocabulary wrapper
            root: image directory
            transform: image transformer
        """
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root = root
        self.img_root = os.path.join(root, 'flickr30k_images')
        self.annotation_root = os.path.join(root, 'dataset_flickr30k.json')
        self.vocab = vocab
        # load the annotations
        with open(self.annotation_root) as f:
            data = json.load(f)
        assert data['dataset'] == 'flickr30k'
        
        # get annotations given the split
        self.annos = [img_anno for img_anno in data['images'] if img_anno['split'] == split]
        
        self.transform = transform
        f.close()
    
    def add_vocab(self, vocab):
        """
        Add vocab to the data loader.
        """
        self.vocab = vocab
    
    def __getitem__(self, index):
        """Returns one data pair"""
        filename = os.path.join(self.img_root, self.annos[index]['filename'])
        
        # load an image
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        # Captions
        caption = self.annos[index]['sentences'][np.random.randint(5)]['raw']  # pick up a random caption from 5 captions
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self.vocab('<start>')]
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)

        return img, target
    
    def __len__(self):
        return len(self.annos)
    
    def __call__(self):
        """Print split, length, image root"""
        print('-------flickr30k--------')
        print('image root:', self.img_root)
        print('dataset split:', self.split)
        print('the length of the dataset:', self.__len__())
    
    def get_img(self, img_id):
        """
        Return an image as numpy array given an image ID.
        """
        filename = self.annos[img_id]['filename']
        img = np.asarray(Image.open(os.path.join(self.img_root, filename)).convert('RGB'))
        return img
    
    def get_captions(self, img_id):
        """
        Return captions of an image given an image ID.
        """
        img_anno = self.annos[img_id]
        captions = [sent['raw'] for sent in img_anno['sentences']]
        return captions

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths
    
def get_loader(root, split, vocab, transform, batch_size=8, shuffle=True, num_workers=4):
    flickr = Flickr30k(split=split, vocab=vocab, root=root, transform=transform)
    
    data_loader = data.DataLoader(dataset=flickr, batch_size=batch_size, 
                                  shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader