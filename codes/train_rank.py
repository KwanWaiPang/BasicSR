import os.path
import sys
import math
import argparse
import time
import random
from collections import OrderedDict

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.logger import Logger, PrintLogger
from utils.rank_test import rank_pair_test


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    parser.add_argument('-fine_tune', '--fine_tune', dest='fine_tune', action='store_true',
                        help='fine tune')
    args = parser.parse_args()
    if not args.fine_tune:
        print('Training from scratch')
        # opt = option.parse(parser.parse_args().opt, is_train=True, fine_tune=False)
        opt = option.parse(parser.parse_args().opt, is_train=True)

    elif args.fine_tune:
        print('Training by fine-tuning')
        opt = option.parse(parser.parse_args().opt, is_train=True, fine_tune=True)
    print(opt['path']['experiments_root'])
    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old experiments if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and \
                 not key == 'pretrain_model_G' and not key == 'pretrain_model_D'))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # print to file and std_out simultaneously
    sys.stdout = PrintLogger(opt['path']['log'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epoches = int(math.ceil(total_iters / train_size))
            print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_dataset_opt = dataset_opt
            val_set = create_dataset(dataset_opt, is_train=False)
            val_loader = create_dataloader(val_set, dataset_opt)
            assert val_loader is not None
            print('Number of val images in [%s]: %d' % (dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
    assert train_loader is not None

    # label file path
    label_path = opt['datasets']['val']['dataroot_label_file']

    # Create model
    model = create_model(opt)

    # create logger
    logger = Logger(opt)

    current_step = 0
    start_time = time.time()
    print('---------- Start training -------------')
    for epoch in range(total_epoches):
        for i, train_data in enumerate(train_loader):

            current_step += 1
            if current_step > total_iters:
                break

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            time_elapsed = time.time() - start_time
            start_time = time.time()

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                for k, v in logs.items():
                    print_rlt[k] = v
                print_rlt['lr'] = model.get_current_learning_rate()
                logger.print_format_results('train', print_rlt)

            # save models
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                print('Saving the model at the end of iter %d' % (current_step))
                model.save(current_step)
                model.test()

            # validation
            if current_step % opt['train']['val_freq'] == 0:

                print('---------- validation -------------')
                start_time = time.time()

                avg_psnr = 0.0
                idx = 0

                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['img1_path'][0]))[0]

                    img_dir = os.path.join(opt['path']['val_images'], str(current_step))
                    util.mkdir(img_dir)
                    f = open(os.path.join(img_dir, 'predict_score.txt'), 'a')

                    model.feed_data(val_data, volatile=True, need_img2=False)
                    model.test()

                    visuals = model.get_current_visuals()
                    predict_score1 = visuals['predict_score1'].numpy()

                    # Save predict scores
                    f.write('%s  %f\n' % (img_name + '.png', predict_score1))
                    f.close()

                # calculate rank accuracy
                # full_accuracy = rank_test(os.path.join(img_dir,'predict_score.txt'),label_path)
                aligned_pair_accuracy, accuracy_esrganbig, accuracy_srganbig = rank_pair_test(
                    os.path.join(img_dir, 'predict_score.txt'), label_path)

                time_elapsed = time.time() - start_time
                # Save to log
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                # print_rlt['full_accuracy'] = full_accuracy
                print_rlt['aligned_pair_accuracy'] = aligned_pair_accuracy
                print_rlt['accuracy_srganbig'] = accuracy_srganbig
                print_rlt['accuracy_esrganbig'] = accuracy_esrganbig
                logger.print_format_results('val', print_rlt)
                print('-----------------------------------')

            # update learning rate
            model.update_learning_rate()

    print('Saving the final model.')
    model.save('latest')
    print('End of Training \t Time taken: %d sec' % (time.time() - start_time))


if __name__ == '__main__':
    # # OpenCV get stuck in transform when used in DataLoader
    # # https://github.com/pytorch/pytorch/issues/1838
    # # However, cause problem reading lmdb
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main()
