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
## DataSet
### Download
We will public all our data soon!

### Introduction

## Citation
