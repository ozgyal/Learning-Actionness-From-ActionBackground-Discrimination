from args import get_args

from featextract.feature import Feature
from featextract.dataset import VideoDataset

from torch.utils.data import DataLoader


def get_dataloader(args, task_steps, task_steps_all):
    dataset = VideoDataset(
        video_list=args.video_list,
        task_steps=task_steps,
        task_steps_all=task_steps_all,
        videos_directory=args.videos_directory,
        fps=args.fps,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        crop_only=False,
        center_crop=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_readers,
    )

    return dataloader


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


def get_all_steps(task_steps):
    all_steps = list()
    for task in task_steps:
        for step in task_steps[task]:
            all_steps.append(step)

    return list(set(all_steps))


if __name__ == '__main__':
    args = get_args()

    task_steps = read_task_info(args.task_list)['steps']  # individual task steps for each task
    task_steps_all = get_all_steps(task_steps)  # all unique steps concatenated

    # Video and text feature extraction
    dataloader = get_dataloader(args, task_steps, task_steps_all)
    feature = Feature(dataloader, args.net_weights_path, args.use_gpu, args.word2vec_path,
                      args.video_features_directory)
    feature.extract_video_features()  # Extracts and saves video features in given directory

    # TODO
    # Get baseline scores/calculations for both test and validation
    # Get auxiliary features
    # Learning actionness model
        # Model training with validation set
        # Combine with actionness score
            # crosstask
            # coin
        # Parameter tuning with holdout

    # Run model on test set

