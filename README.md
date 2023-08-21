# LiveChat: A Large-Scale Personalized Dialogue Dataset Automatically Constructed from Live Streaming
This is the official repository for the ACL 2023 paper "LiveChat: A Large-Scale Personalized Dialogue Dataset Automatically Constructed from Live Streaming"

![DataConstruction](./Image/DataConstruction.png)
LiveChat is a large-scale dataset, composed of 1.33 million real-life Chinese dialogues with almost 3800 average sessions across 351 personas and fine-grained profiles for each persona. LiveChat is automatically constructed by processing numerous live videos on the Internet and naturally falls within the scope of multi-party conversations.

This repo implements two benchmark tasks (Response Modeling and Addressee Recognition) and a generation task (Generation) for LiveChat:

- [Response Modeling](https://github.com/gaojingsheng/LiveChat/tree/master/Tasks/ResponseModeling) (Retrival-based) 
- [Addressee Recognition](https://github.com/gaojingsheng/LiveChat/tree/master/Tasks/AddresseeRecognition)
- [Generation](https://github.com/gaojingsheng/LiveChat/tree/master/Tasks/Generation) (BART)

Instructions of how to run these models on the two tasks are described in their README files. Before trying them, you need to first download the dataset and unzip it into the folder ./Dataset. The file tree should be like

```
.
+-- dataset
|   +-- train.json
|   +-- val.json
|   +-- test.json
|   +-- basic_profile.json
|   +-- text_profile.json
```

## Enviroment
You need to clone our project：
```bash
$ git clone https://github.com/gaojingsheng/LiveChat.git
```


Create the environment and download the packages
```bash
$ conda create -n LiveChat python==3.8
$ conda activate LiveChat
$ pip install -r requirements.txt
```

## DataSet
### Download
Please refer to dataset [README.md](https://github.com/gaojingsheng/LiveChat/blob/master/Dataset/README.md)


## Citation
If you find our paper and repository useful, please cite us in your paper:
```
@inproceedings{gao-etal-2023-livechat,
    title = "{L}ive{C}hat: A Large-Scale Personalized Dialogue Dataset Automatically Constructed from Live Streaming",
    author = "Gao, Jingsheng  and
      Lian, Yixin  and
      Zhou, Ziyi  and
      Fu, Yuzhuo  and
      Wang, Baoyuan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.858",
    doi = "10.18653/v1/2023.acl-long.858",
    pages = "15387--15405",
}
```
