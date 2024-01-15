# Learning-Actionness-From-ActionBackground-Discrimination
PyTorch implementation of the paper ["Learning actionness from action/background discrimination"](https://link.springer.com/article/10.1007/s11760-022-02369-y) for action localization 
on the [CrossTask](https://github.com/DmZhukov/CrossTask) dataset. Tested with Python 3.8.13, PyTorch 1.11.0, 
Numpy 1.22.4, [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) 0.2.0 .

## 1. Feature Extraction

For video and text feature extraction [MIL-NCE](https://github.com/antoine77340/MIL-NCE_HowTo100M/tree/master) is used.
- First, follow the link in [MIL-NCE](https://github.com/antoine77340/MIL-NCE_HowTo100M/tree/master) and download the 
word2vec matrix and dictionary. Arrange the *--word2vec_path* and *--dict_path* arguments in 
args.py accordingly. 
- Then, download the pretrained S3D weights *"s3d_howto100m.pth"* from [S3D](https://github.com/antoine77340/S3D_HowTo100M)
and update the *--net_weights_path*.
- Follow the instructions given in [CrossTask](https://github.com/DmZhukov/CrossTask)
to download the videos. Update the *--videos_path* and *--annotations_path*.
- Arrange *--video_features_path* and *--text_features_path*.
- Run extract_features.py . In order to use different videos, you should update the featextract/data/video_list.csv .

## 2. Getting the Baseline Scores

This part is for replacating the action localization result given in [MIL-NCE](https://github.com/antoine77340/MIL-NCE_HowTo100M/tree/master)
by using the previously extracted features.

Repo will be updated regularly with further implementation of the paper.
