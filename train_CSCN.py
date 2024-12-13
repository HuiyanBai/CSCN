import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from dataset import *
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
from torch.utils.tensorboard import SummaryWriter
from args import get_args
from CSCN import CSCN
from utils import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def main():

    print('Build model ...')

    args = get_args()
    set_seed(seed=args.seed)
    model_name = args.model_name
    config = args.config
    epoch_num = args.epoch_num
    batch_size = args.BatchSize

    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    exp = date_time

    model = CSCN(config).to(device)

    print('Load data ...')

    data_root = args.data_root

    train_image_list, train_label_list, test_image_list, test_label_list = get_image(data_root)

    def _init_fn(worker_id):
        np.random.seed(int(args.seed) + worker_id)
    train_dataset = WHU_OHS_Dataset(image_file_list=train_image_list, label_file_list=train_label_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                              worker_init_fn=_init_fn)

    test_dataset = WHU_OHS_Dataset(image_file_list=test_image_list, label_file_list=test_label_list)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4,
                             worker_init_fn=_init_fn)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    lambda_lr = lambda x: (1 - x / epoch_num) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    save_dir = args.save_dir
    model_path = save_dir + 'model/' + model_name + '_' + \
                 'Epoch' + '_' + str(epoch_num) + '+' + 'Batchsize' + '_' + str(batch_size) + '_' + exp + '/'
    image_path = save_dir + 'images/' + model_name + '_' + \
                 'Epoch' + '_' + str(epoch_num) + '+' + 'Batchsize' + '_' + str(batch_size) + '_' + exp + '/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    print('Start training.')
    print(model_path)
    max_OA = 0
    max_mean_F1 = 0
    train_OA, test_OA = 0, 0
    class_num = 24
    epoch_max_OA, epoch_max_mean_F1 = 0, 0

    writer_loss = SummaryWriter(save_dir + 'tensorboard/loss/' + model_name + '_' + exp + '/')

    for epoch in range(epoch_num):
        print('Epoch: %d/%d' % (epoch + 1, epoch_num))
        model.train()

        batch_index = 1
        loss_sum_total = 0
        average_loss_total = 0

        for data, data_diff, label, _ in tqdm(train_loader):

            data = data.to(device)
            data_diff = data_diff.to(device)

            label = label.to(device)

            optimizer.zero_grad()

            pred, loss_uff = model(data=data, data_spec=data_diff, label=label, training=True)

            loss_CrossEntropy = criterion(pred, label)

            loss_total = loss_CrossEntropy + loss_uff

            loss_total.backward()

            optimizer.step()

            loss_sum_total = loss_sum_total + loss_total.item()
            batch_index = batch_index + 1
            average_loss_total = loss_sum_total / batch_index

        writer_loss.add_scalar('total_loss', average_loss_total, epoch + 1)

        ''' ---------------------------------------- Test ---------------------------------------- '''

        with torch.no_grad():
            model.eval()
            confusionmat_test = torch.zeros([class_num, class_num]).to(device)
            count = 0

            for data, data_diff, label, _ in tqdm(test_loader):

                data = data.to(device)
                data_diff = data_diff.to(device)

                label = label.to(device)

                count = count + 1

                pred, _ = model(data=data, data_spec=data_diff, label=label, training=False)

                output_test = pred[0, :, :, :].argmax(axis=0)
                label_test = label[0, :, :]

                confusionmat_tmp_test = genConfusionMatrix(class_num, output_test, label_test)
                confusionmat_test = confusionmat_test + confusionmat_tmp_test

        confusionmat_test = confusionmat_test.cpu().detach().numpy()
        unique_index_test = np.where(np.sum(confusionmat_test, axis=1) != 0)[0]
        confusionmat_test = confusionmat_test[unique_index_test, :]
        confusionmat_test = confusionmat_test[:, unique_index_test]

        a_test = np.diag(confusionmat_test)
        b_test = np.sum(confusionmat_test, axis=0)
        c_test = np.sum(confusionmat_test, axis=1)

        eps = 0.0000001

        PA = a_test / (c_test + eps)
        UA = a_test / (b_test + eps)
        F1 = 2 * PA * UA / (PA + UA + eps)
        mean_F1 = np.nanmean(F1)
        test_OA = np.sum(a_test) / np.sum(confusionmat_test)

        def log_string(str):
            logger.info(str)
            print(str)

        path = Path.cwd()
        logging_dir = path.joinpath(save_dir + 'logging/')
        logging_dir.mkdir(exist_ok=True)
        name_1 = model_path.split('/')[-2]
        name_2 = model_path.split('/')[-1]
        txt_path = save_dir + 'logging/' + 'save_data' + '_train_' + name_1 + '_' + name_2 + '.txt'

        # ===============  这里 亲测有效！！ ==================
        while len(logging.root.handlers) > 0:
            logging.root.handlers.pop()
        # ==================================================

        logger = logging.getLogger("Model")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(txt_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if (test_OA > max_OA):
            max_OA = test_OA
            epoch_max_OA = epoch + 1
            torch.save(model.state_dict(), model_path + model_name + '_update_OA' + '.pth')
            log_string('-----------------------------------------   OA    ----------------------------------------')

        if (mean_F1 > max_mean_F1):
            max_mean_F1 = mean_F1
            epoch_max_mean_F1 = epoch + 1
            torch.save(model.state_dict(), model_path + model_name + '_update_mean_F1' + '.pth')
            log_string('----------------------------------------- mean_F1 ----------------------------------------')

        log_string('Epoch:[{}/{}], Learning rate={:.9f}, loss_total={:.6f}\n'
                   'train OA={:.6f}, test OA={:.6f}, max_test OA={:.6f}, Epoch_max_OA={}, mean_F1={:.6f}, '
                   'max_mean_F1={:.6f}, Epoch_max_mean_F1={}'
                   .format(epoch+1, epoch_num, optimizer.state_dict()['param_groups'][0]['lr'], average_loss_total,
                           train_OA, test_OA, max_OA, epoch_max_OA,
                           mean_F1, max_mean_F1, epoch_max_mean_F1))

        scheduler.step()

    # Save model for the final epoch
    torch.save(model.state_dict(), model_path + model_name + '_final.pth')
    # ===============  这里 亲测有效！！ ==================
    while len(logging.root.handlers) > 0:
        logging.root.handlers.pop()
    # ==================================================
    writer_loss.close()
    logger.handlers.clear()
    logger.info('finish training!')
    args = get_args()
    f = open(args.save_dir + 'result/result_Encoder.txt', "a+")
    f.write('\n' + model_name + '---max_mean_F1:  ' + str(max_mean_F1) + '\n')


if __name__ == '__main__':
    main()



