from args import get_args

from featextract.video_feature import VideoFeature
from featextract.video_dataset import VideoDataset

from featextract.text_feature import TextFeature
from featextract.text_dataset import TextDataset

from torch.utils.data import DataLoader


def read_task_info(task_list):  # this code piece is taken from: https://github.com/DmZhukov/CrossTask
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(task_list, 'r') as f:
        idx = f.readline()
        while idx != '':
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(',')
            next(f)
            idx = f.readline()
    return {'title': titles, 'url': urls, 'n_steps': n_steps, 'steps': steps}


def get_all_unique_steps(steps):
    all_steps = list()
    for task in steps:
        for step in steps[task]:
            if step not in all_steps:
                all_steps.append(step)

    return all_steps


def get_dataloader(args, dataset):

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_readers,
    )

    return dataloader


def get_video_dataset(args):
    dataset = VideoDataset(
        video_list=args.video_list,
        videos_path=args.videos_path,
        fps=args.fps,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        crop_only=False,
        center_crop=True,
    )

    return dataset


def get_video_feature(args, dataloader):
    feature = VideoFeature(dataloader, args.net_weights_path, args.use_gpu, args.word2vec_path,
                                        args.video_features_path)

    return feature


def get_text_dataset(args, steps):
    dataset = TextDataset(steps, args.token_to_word_path, max_words=4)

    return dataset


def get_text_feature(args, dataloader,):
    feature = TextFeature(dataloader, args.net_weights_path, args.use_gpu, args.word2vec_path, args.text_features_path)

    return feature


if __name__ == '__main__':
    all_arguments = get_args()

    # Video feature extraction
    video_dataset = get_video_dataset(all_arguments)
    video_dataloader = get_dataloader(all_arguments, video_dataset)
    video_feature = get_video_feature(all_arguments, video_dataloader)
    # video_feature.extract_features()  # Extracts and saves video features into the given directory

    # Text feature extraction
    task_based_steps = read_task_info(all_arguments.task_list)['steps']  # steps for each task
    steps = get_all_unique_steps(task_based_steps)  # all unique steps concatenated
    text_dataset = get_text_dataset(all_arguments, steps)
    text_dataloader = get_dataloader(all_arguments, text_dataset)
    text_feature = get_text_feature(all_arguments, text_dataloader)
    text_feature.extract_features()  # Extracts and saves step name features into the given directory



