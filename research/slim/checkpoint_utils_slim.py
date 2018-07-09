import os
import collections


def get_checkpoint_list(checkpoints_dir):
    checkpoints_list = [checkpoint.replace('.meta', '') for checkpoint in os.listdir(checkpoints_dir) if
                        all(string in checkpoint for string in ['model.ckpt', 'meta'])]
    checkpoints_list = sorted(checkpoints_list, key=lambda x: float(x.replace('model.ckpt-', '')))
    checkpoints_list = [os.path.join(checkpoints_dir, checkpoint) for checkpoint in checkpoints_list]
    return checkpoints_list


if __name__ == "__main__":
    checkpoints_list = get_checkpoint_list(
        "/media/razor30/Large/PycharmProjects/mantis-slim/razor_segmentation/training_outputs/mantis_check_softmax")
    pass
