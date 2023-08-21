# Processed Dataset for LiveChat

The tree structure of the process dataset is as follows:
```
.
+-- ProcessedData
|   +-- 400kDialogueInPaper
|       +-- train_data.pk
|       +-- dev_data.pk
|   +-- AddresseeRecognition
|       +-- train_data.pk
|       +-- dev_data.pk
|       +-- test_data.pk
|   +-- RawDialogueData
|       +-- train_data.pk
|       +-- dev_data.pk
|       +-- test_data.pk
|   +-- basic_profile.json
|   +-- text_profile.json

```
## 400kDialogueinPaper

This includes the training (train_data.pk) and testing (dev_data.pk) sets of the dialogue dataset used in the paper.

each dialogue sample is composed by a list:
```
[streamer_id, audience_comment, streamer_reponse]
```

## AddresseeRecognition

This includes the training (train_data.pk), deving (dev_data.pk) and testing (test_data.pk) sets of the addressee recognition dataset used in the paper.

each sample is composed by a list:
```
[streamer_id, audience_comment_list, streamer_reponse]
```
The length of each audience_comment_list is 10, where the last sencetence is the target addresse of the streamer_response.

## RawDialogueData

This includes the raw all training (train_data.pk), deving (dev_data.pk) and testing (test_data.pk) sets of the dialogue dataset of LiveChat.

each sample is composed by a list:
```
[streamer_id, audience_comment, streamer_reponse]
```

## basic_profile.json

This includes the basic profile of the streamers, where some characteristics of the streamers are replaced by anonymous numbers.

## text_profile.json

This includes the text profile (max_length is 512) of the streamers used in our paper. 

