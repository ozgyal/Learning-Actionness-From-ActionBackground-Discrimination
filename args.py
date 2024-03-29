import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_list', type=str, default='featextract/data/tasks_primary.txt')
    parser.add_argument('--video_list', type=str, default='featextract/data/example_video_list.csv')
    parser.add_argument('--videos_path', type=str, default='/Users/ozge/PycharmProjects/crosstask_data/videos')
    parser.add_argument('--net_weights_path', type=str, default='/Users/ozge/PycharmProjects/s3d_howto100m.pth')
    parser.add_argument('--word2vec_path', type=str, default='/Users/ozge/PycharmProjects/word2vec.pth')
    parser.add_argument('--dict_path', type=str, default='/Users/ozge/PycharmProjects/dict.npy')
    parser.add_argument('--video_features_path', type=str,
                        default='/Users/ozge/PycharmProjects/crosstask_data/video_features')
    parser.add_argument('--text_features_path', type=str,
                        default='/Users/ozge/PycharmProjects/crosstask_data/text_features')
    parser.add_argument('--annotations_path', type=str,
                        default='/Users/ozge/PycharmProjects/crosstask_data/annotations')
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--frame_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_thread_readers', type=int, default=1)
    parser.add_argument('--use_gpu', type=int, default=0)

    args = parser.parse_args()

    return args
