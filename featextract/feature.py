import torch as th
import os.path

from featextract.s3dg import S3D


class Feature:
    def __init__(self, dataloader, net_weights_path, use_gpu, word2vec_path, video_features_directory):
        self.dataloader = dataloader
        self.net_weights_path = net_weights_path
        self.use_gpu = use_gpu
        self.word2vec_path = word2vec_path
        self.video_features_directory = video_features_directory
        self.net = self._set_network()

    def _set_network(self):
        # Initiate the model
        net = S3D(space_to_depth=True, word2vec_path=self.word2vec_path)

        # Load the model weights
        net.load_state_dict(th.load(self.net_weights_path))

        if self.use_gpu:
            net.cuda()

        net = net.eval()

        return net

    def extract_video_features(self):

        with th.no_grad():
            for i_batch, data in enumerate(self.dataloader):
                video = data['video'].float()
                video_id = data['video_id']

                video = video / 255.0
                video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])

                # For each clip in a video, get embeddings separately
                clip_feature_list = []
                for i in range(0, video.shape[0]):
                    if self.use_gpu:
                        clip_feature = self.net.forward_video(
                            video[i, :, :, :, :].view(1, video.shape[1], video.shape[2], video.shape[3],
                                                      video.shape[4]).cuda(), mixed5c=True)
                    else:
                        clip_feature = self.net.forward_video(
                            video[i, :, :, :, :].view(1, video.shape[1], video.shape[2], video.shape[3],
                                                      video.shape[4]), mixed5c=True)

                    clip_feature_list.append(clip_feature)

                # Concatenate clip features
                video_feature = th.cat(clip_feature_list)
                th.save(video_feature, os.path.join(self.video_features_directory, video_id[0]))
                print(video_id[0] + ' done')
