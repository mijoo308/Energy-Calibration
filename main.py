import sys
import argparse
import os

import torch
import pickle
import numpy as np

import CONFIG
sys.path.append(".")


from source.model_factory import get_network, inference
from source.data_utils import get_dataloader, get_all_severity_dataloader
from source.Evaluation import Evaluation
from source.calibration import *
from source.utils import ece_result
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Based Instance-wise Calibration")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the dataloader.')
    parser.add_argument('--net', type=str, default='densenet201', help='Model architecture (e.g., resnet50).')
    parser.add_argument('--weight_path', type=str, default='./weights/densenet201_cifar10.pth', help="Path to the pre-trained model weights.")
    parser.add_argument('--gpu', action='store_true', default=False, help='Flag to use GPU if available.')
    parser.add_argument('--data', type=str, default='cifar10', help='Dataset name (e.g., cifar10, cifar100, imagenet).')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for the dataset.')
    parser.add_argument('--num_worker', type=int, default=0, help='Number of workers for data loading (set to 0 for Windows).')
    parser.add_argument('--ddp', action='store_true', default=False, help='Flag indicating if the model was trained in a DDP (Distributed Data Parallel) environment.')
    parser.add_argument('--save_dir', type=str, default='./result', help='Directory to save the results.')
    parser.add_argument('--ood_train', type=str, default='SVHN', help='Semantic OOD dataset name for tuning (e.g., SVHN).')
    parser.add_argument('--ood_test', type=str, default='texture', help='Semantic OOD dataset name for testing (e.g., texture).')

    # Convenience flags
    parser.add_argument('--id_train_inferenced', type=bool, default=False, help='already have all logit files for tuning')
    parser.add_argument('--ood_train_inferenced', type=bool, default=False, help='already have all ood logit files for tuning')
    parser.add_argument('--test_inferenced', type=bool, default=False, help='already have all test logit files for evaluation')

    args = parser.parse_args()

    torch.manual_seed(512)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, args.data)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, args.net)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    '''[ (0) check GPU setting ]'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.gpu:
        if device == 'cpu':
            print('CUDA is not available')
            sys.exit()
        print('[GPU INFO]')
        print(torch.cuda.memory_summary(), end='')


    '''[ (1) Load pre-trained model ]'''
    net = get_network(network_name=args.net, dataset=args.data, path=args.weight_path, ddp_trained=args.ddp)
    net.to(device)
    net.eval()

    '''[ (2) Prepare logits for ID valset & semantic OOD] '''
    val_logits, val_labels = None, None
    ood_logits, ood_labels = None, None

    '''(2-1) Prepare logits for ID valset '''
    if args.id_train_inferenced:
        val_logit_path = os.path.join(save_dir, f'{args.net}_{args.data}_val_logit.pkl')
        val_label_path = os.path.join(save_dir, f'{args.net}_{args.data}_val_label.pkl')
        assert os.path.isfile(val_logit_path), 'No saved val logit file'
        assert os.path.isfile(val_label_path), 'No saved val label file'
        print('found ID val logit files ... ')

        with open(val_logit_path, 'rb') as f:
            val_logits = pickle.load(f)
        with open(val_label_path, 'rb') as f:
            val_labels = pickle.load(f)

    else:
        val_dataloader = get_dataloader(
            root = args.data_root,
            data_name=args.data,
            batch_size=args.batch_size,
            num_workers=args.num_worker,
            eval=False
        )

        val_logits, val_labels = inference(net, val_dataloader, gpu=args.gpu)
        with open(os.path.join(save_dir, f'{args.net}_{args.data}_val_logit.pkl'), 'wb') as f:
            pickle.dump(val_logits, f)
        with open(os.path.join(save_dir, f'{args.net}_{args.data}_val_label.pkl'), 'wb') as f:
            pickle.dump(val_labels, f)
    

    if args.data == 'cifar10':
        n_classes, n_sem = 10, CONFIG.CIFAR10_NUM_SEM
    elif args.data == 'cifar100':
        n_classes, n_sem = 100, CONFIG.CIFAR100_NUM_SEM
    elif args.data == 'imagenet':
        n_classes, n_sem = 1000, CONFIG.IMAGENET_NUM_SEM


    '''(2-2) Prepare logits for semantic OOD '''

    if args.ood_train_inferenced:
        ood_logit_path = os.path.join(save_dir, f'{args.net}_{args.ood_train}_ood_train_logit.pkl')
        ood_label_path = os.path.join(save_dir, f'{args.net}_{args.ood_train}_ood_train_label.pkl')
        assert os.path.isfile(ood_logit_path), 'No saved ood train logit file'
        assert os.path.isfile(ood_label_path), 'No saved ood train label file'
        print('found semantic ood train logit files ... ')

        with open(ood_logit_path, 'rb') as f:
            ood_logits = pickle.load(f)
        with open(ood_label_path, 'rb') as f:
            ood_labels = pickle.load(f)

    else:
        ood_train_dataloader = get_dataloader(
            root = args.data_root,
            data_name=args.ood_train,
            batch_size=args.batch_size,
            num_workers=args.num_worker,
            eval=False,
            num_sample=n_sem
        )

        ood_logits, ood_labels = inference(net, ood_train_dataloader, gpu=args.gpu)
        with open(os.path.join(save_dir, f'{args.net}_{args.ood_train}_ood_train_logit.pkl'), 'wb') as f:
            pickle.dump(ood_logits, f)
        with open(os.path.join(save_dir, f'{args.net}_{args.ood_train}_ood_train_label.pkl'), 'wb') as f:
            pickle.dump(ood_labels, f)

    val_logits = val_logits.cpu().numpy()
    val_labels = val_labels.cpu().numpy()

    ood_logits = ood_logits.cpu().numpy()
    ood_labels = ood_labels.cpu().numpy()


    '''[ (3) Tune post-hoc calibrator ] '''
    t = train_temperature_scaling(val_logits, val_labels, loss='ce')
    w = train_ensemble_scaling(val_logits, val_labels, t, n_classes, loss='mse')
    irm = train_isotonic_regression(val_logits, val_labels)
    irm_list = train_irova(val_logits, val_labels)
    irovats_t, list_ir = train_irovats(val_logits, val_labels, loss='mse')
    SPL_frecal, p_wo_DAC, label_wo_DAC = train_spline(val_logits, val_labels)
    theta, pdf_correct, pdf_incorrect = train_energycal(val_logits, val_labels, ood_logits, t)
    
    
    '''[ (4) Test & Evaluation ]'''
    # ID test data + corrupted data (covariate OOD)
    all_test_dataloaders = get_all_severity_dataloader(
        root = args.data_root,
        max_corrupt_level=5,
        data_name=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        eval=True
    )

    # semantic OOD data
    ood_test_dataloader = get_dataloader(
        root = args.data_root,
        data_name=args.ood_test,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        eval=True
    )

    # [level0(ID), level1, level2, level3, level4, level5, level6(semantic OOD)]
    all_test_dataloaders.append(ood_test_dataloader)

    all_level_results = []
    for level in range(len(all_test_dataloaders)):
        for c_type in CONFIG.CORRUPTION_TYPE:

            result = {} # {'ACC':{}, 'ECE':{} , ...}
            for metric in CONFIG.EVAL_METRIC:
                result[metric] = {}

            if level == 0 or level == 6: # 0 : ID /  6 : sem OOD
                c_type = None
                test_dataloader = all_test_dataloaders[level]
            else:                       # 1~5 : corrupted 
                test_dataloader = all_test_dataloaders[level][c_type]

            ''' (4-1) Prepare test logit '''
            test_logit_path = os.path.join(save_dir, f'{args.net}_{args.data}_{c_type}_{level}_test_logit.pkl')
            test_label_path = os.path.join(save_dir, f'{args.net}_{args.data}_{c_type}_{level}_test_label.pkl')

            test_logits, test_labels = None, None
            if args.test_inferenced:
                assert os.path.isfile(test_logit_path), 'No saved test logit file'
                assert os.path.isfile(test_label_path), 'No saved test label file'

                print(f'found [severity {level} : {c_type}] test logit files ... ')
                with open(test_logit_path, 'rb') as f:
                    test_logits = pickle.load(f)
                with open(test_label_path, 'rb') as f:
                    test_labels = pickle.load(f)

            else:
                print(f'Inferencing [severity {level} : {c_type}] ... ')
                test_logits, test_labels = inference(net, test_dataloader, gpu=args.gpu)

                if level == 6:
                    test_labels = torch.zeros((test_logits.shape[0],))

                with open(test_logit_path, 'wb') as f:
                    pickle.dump(test_logits, f)
                with open(test_label_path, 'wb') as f:
                    pickle.dump(test_labels, f)

            test_logits = test_logits.cpu()
            test_labels = test_labels.cpu()
            

            ''' (4-2) Apply Post-Hoc Calibration '''
            uncal_probs = softmax(test_logits, 1)
            calib_prob_ts = calibrate_ts(test_logits, t)
            calib_prob_ets = calibrate_ets(test_logits, w, t, n_classes)
            calib_probs_mir = calibrate_isotonic_regression(test_logits, irm)
            calib_probs_irova = calibrate_irova(test_logits, irm_list)
            calib_probs_irovats = calibrate_irovats(test_logits, irovats_t, list_ir)
            calib_probs_spline, tacc = calibrate_spline(SPL_frecal, test_logits, test_labels)
            calib_probs_ebs = calibrate_energycal(test_logits, t, theta, pdf_correct, pdf_incorrect)


            ''' (4-3) Evaluation after calibration '''
            ALL_METHOD_RESULT = {}
            uncal_eval = Evaluation(uncal_probs, test_labels)
            ts_eval = Evaluation(calib_prob_ts, test_labels)
            ets_eval = Evaluation(calib_prob_ets, test_labels)
            irm_eval = Evaluation(calib_probs_mir, test_labels)
            irova_eval = Evaluation(calib_probs_irova, test_labels)
            irovats_eval = Evaluation(calib_probs_irovats, test_labels)
            spline_eval = Evaluation(calib_probs_spline, test_labels)
            ours_eval = Evaluation(calib_probs_ebs, test_labels)


            ALL_METHOD_RESULT['uncal'] = uncal_eval
            ALL_METHOD_RESULT['TS'] = ts_eval
            ALL_METHOD_RESULT['ETS'] = ets_eval
            ALL_METHOD_RESULT['IRM'] = irm_eval
            ALL_METHOD_RESULT['IROVA'] = irova_eval
            ALL_METHOD_RESULT['IROVATS'] = irovats_eval
            ALL_METHOD_RESULT['SPLINE'] = spline_eval
            ALL_METHOD_RESULT['Ours'] = ours_eval

            for method in ALL_METHOD_RESULT.keys():
                for metric in CONFIG.EVAL_METRIC:
                    result[metric][method] = getattr(ALL_METHOD_RESULT[method], metric)()

            # [level] 0 : indomain / 1~5: corrupted OOD / 6 : sem OOD
            with open(os.path.join(save_dir, f'{args.net}_{args.data}_{c_type}_{level}_result.pkl'), 'wb') as f:
                pickle.dump(result, f)

            if level == 0 or level == 6:
                break
            
    
    ''' [5] Calibration Result '''
    res = ece_result(network=args.net, dataset=args.data,
               max_level=6, result_dir=save_dir,
               calib_methods=['uncal', 'TS', 'ETS', 'IRM', 'IROVA', 'IROVATS', 'SPLINE', 'Ours'])
    
    print(f'[{args.net}]')
    print(tabulate(res, headers='keys', tablefmt='pretty'))
