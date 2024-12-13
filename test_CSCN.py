import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
from args import get_args
from CSCN import CSCN
import cv2
from confusionmatrix import *
from utils import *


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def main():
    print('Build model ...')

    args = get_args()
    set_seed(seed=args.seed)
    config = args.config

    model = CSCN(config).to(device)
    # model_path = '/model/CSCN.pth'
    model_path = ("/data/users/baihuiyan/DATASET/Experiment_FreeNet/experiment_debug/model/"
                  "CSCN_Epoch_50+Batchsize_2_2024_12_13_13_44_49/CSCN_update_OA.pth")
    model.load_state_dict(torch.load(model_path))

    print('Loaded trained model.')

    print('Load data ...')

    data_root = args.data_root

    _, _, test_image_list, test_label_list = get_image(data_root)

    class_num = 24

    def _init_fn(worker_id):
        np.random.seed(int(args.seed) + worker_id)
    test_dataset = WHU_OHS_Dataset(image_file_list=test_image_list, label_file_list=test_label_list)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2,
                             worker_init_fn=_init_fn)

    labels_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                   '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
    drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name)

    print('Testing.')
    save_image_path = 'result/' + model_path.split('.')[0].split('/')[-2] + '/'

    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    with torch.no_grad():
        model.eval()
        total_time = []
        confusionmat = torch.zeros([class_num, class_num]).to(device)

        for data, data_diff, label, name in tqdm(test_loader):

            data = data.to(device)
            data_diff = data_diff.to(device)

            label = label.to(device)

            pred, _ = model(data=data, data_spec=data_diff, label=label, training=False)

            output = pred[0, :, :, :].argmax(axis=0)
            label = label[0, :, :]

            confusionmat_tmp = genConfusionMatrix(class_num, output, label)
            confusionmat = confusionmat + confusionmat_tmp

            # visualization
            output_vis = output.detach().cpu().numpy() + 1
            output_vis = output_vis.astype(np.uint8)
            label_vis = label.detach().cpu().numpy() + 1
            output_vis_color = shangse(output_vis, 25, label_vis)
            cv2.imwrite(save_image_path + name[0].split('.')[0] + '.png', output_vis_color)


    confusionmat = confusionmat.cpu().detach().numpy()

    unique_index = np.where(np.sum(confusionmat, axis=1) != 0)[0]
    confusionmat = confusionmat[unique_index, :]
    confusionmat = confusionmat[:, unique_index]

    a = np.diag(confusionmat)
    b = np.sum(confusionmat, axis=0)
    c = np.sum(confusionmat, axis=1)

    eps = 0.0000001

    PA = a / (c + eps)
    dic_PA = dict(map(lambda x, y: [x, y], unique_index + 1, PA))
    UA = a / (b + eps)

    F1 = 2 * PA * UA / (PA + UA + eps)

    mean_F1 = np.nanmean(F1)

    OA = np.sum(a) / np.sum(confusionmat)

    PE = np.sum(b * c) / (np.sum(c) * np.sum(c))
    Kappa = (OA - PE) / (1 - PE)

    intersection = np.diag(confusionmat)
    union = np.sum(confusionmat, axis=1) + np.sum(confusionmat, axis=0) - np.diag(confusionmat)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)

    total_time = np.array(total_time)
    avg_time = np.mean(total_time)
    fps = 1 / avg_time

    print("fps: ", fps)


    def log_string(str):
        logger.info(str)
        print(str)

    path = Path.cwd()
    logging_dir = path.joinpath('/logging/')
    logging_dir.mkdir(exist_ok=True)
    model_name = model_path.split('.')[0]
    name = model_name.split('/')[-2]
    txt_path = '/data/dataset/BHY/Experiment_FreeNet/experiment_20240229/logging/' + 'save_data_test_' + name + '.txt'

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(txt_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PA={}\n UA={}\n F1={}\n mean_F1={}\n OA={}\n Kappa={}\n IoU={}\n mIoU={}\n'.
               format(dic_PA, UA, F1, mean_F1, OA, Kappa, IoU, mIoU))

    # 绘制混淆矩阵
    ConfusionMatrix_name = name
    ConfusionMatrix_path = '/ConfusionMatrix/ConfusionMatrix_' + ConfusionMatrix_name
    drawconfusionmatrix.drawMatrix(path=ConfusionMatrix_path, acc=mean_F1, name=ConfusionMatrix_name)
    print('------------------ ConfusionMatrix ------------------')


if __name__ == '__main__':
    main()
