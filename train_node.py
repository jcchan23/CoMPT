import argparse
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.model_selection import train_test_split
from dataset_node import construct_dataset, mol_collate_func
from transformer_node import make_model
from utils import ScheduledOptim, get_options


def loss_function(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    y_mask = torch.where(y_true != 0., torch.full_like(y_true, 1), torch.full_like(y_true, 0))
    loss = torch.sum(torch.abs(y_true - y_pred * y_mask)) / torch.sum(y_mask)
    return loss


def model_train(model, train_dataset, valid_dataset, model_params, train_params, dataset_name, element):

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    # build optimizer
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_steps'])

    best_valid_loss = float('inf')
    best_epoch = -1
    best_valid_result = dict()

    for epoch in range(train_params['total_epochs']):
        # train
        train_loss = list()
        model.train()
        for batch in tqdm(train_loader):
            adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch_size, max_length, max_length)
            node_features = node_features.to(train_params['device'])  # (batch_size, max_length, d_node)
            edge_features = edge_features.to(train_params['device'])  # (batch_size, max_length, max_length, d_edge)
            y_true = y_true.to(train_params['device'])                      # (batch_size, max_length, 1)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0   # (batch_size, max_length)
            # (batch_size, max_length, 1)
            y_pred = model(node_features, batch_mask, adjacency_matrix, edge_features)
            loss = loss_function(y_true, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            train_loss.append(loss.detach().item())

        # valid
        model.eval()
        with torch.no_grad():
            valid_result = dict()
            valid_result['label'], valid_result['prediction'], valid_result['loss'] = list(), list(), list()
            for batch in tqdm(valid_loader):
                adjacency_matrix, node_features, edge_features, y_true = batch
                adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch_size, max_length, max_length)
                node_features = node_features.to(train_params['device'])  # (batch_size, max_length, d_node)
                edge_features = edge_features.to(train_params['device'])  # (batch_size, max_length, max_length, d_edge)

                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch_size, max_length)
                # (batch_size, max_length, 1)
                y_pred = model(node_features, batch_mask, adjacency_matrix, edge_features)

                y_true = y_true.numpy().flatten()
                y_pred = y_pred.cpu().detach().numpy().flatten()
                y_mask = np.where(y_true != 0., 1, 0)

                times = 0
                for true, pred in zip(y_true, y_pred):
                    if true != 0.:
                        times += 1
                        valid_result['label'].append(true)
                        valid_result['prediction'].append(pred)
                        valid_result['loss'].append(np.abs(true - pred))
                assert times == np.sum(y_mask)

            valid_result['r2'] = metrics.r2_score(valid_result['label'], valid_result['prediction'])

        print('Epoch {}, learning rate {:.6f}, train loss: {:.4f}, valid loss: {:.4f}, valid r2: {:.4f}'.format(
            epoch + 1, optimizer.view_lr(), np.mean(train_loss), np.mean(valid_result['loss']), valid_result['r2']
        ))

        # save the model and valid result
        if np.mean(valid_result['loss']) < best_valid_loss:
            best_valid_loss = np.mean(valid_result['loss'])
            best_epoch = epoch + 1
            best_valid_result = valid_result
            torch.save({'state_dict': model.state_dict(),
                        'best_epoch': best_epoch, 'best_valid_loss': best_valid_loss},
                       f'./Model/{dataset_name}/best_model_{dataset_name}_{element}.pt')

        # temp test
        if (epoch + 1) % 10 == 0:
            checkpoint = torch.load(f'./Model/{dataset_name}/best_model_{dataset_name}_{element}.pt')
            print('=' * 20 + ' middle test ' + '=' * 20)
            test_result = model_test(checkpoint, test_dataset, model_params, train_params)
            print("best epoch: {}, best valid loss: {:.4f}, test loss: {:.4f}, test r2: {:.4f}".format(
                checkpoint['best_epoch'], checkpoint['best_valid_loss'], np.mean(test_result['loss']), test_result['r2']
            ))
            print('=' * 40)

        # early stop
        if abs(best_epoch - epoch) >= 20:
            print("=" * 20 + ' early stop ' + "=" * 20)
            break

    return best_valid_result


def model_test(checkpoint, test_dataset, model_params, train_params):
    # build loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func,
                             shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    # build model
    model = make_model(**model_params)
    model.to(train_params['device'])
    model.load_state_dict(checkpoint['state_dict'])

    # test
    model.eval()
    with torch.no_grad():
        test_result = dict()
        test_result['label'], test_result['prediction'], test_result['loss'] = list(), list(), list()
        for batch in tqdm(test_loader):
            adjacency_matrix, node_features, edge_features, y_true = batch
            adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch_size, max_length, max_length)
            node_features = node_features.to(train_params['device'])  # (batch_size, max_length, d_node)
            edge_features = edge_features.to(train_params['device'])  # (batch_size, max_length, max_length, d_edge)

            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch_size, max_length)
            # (batch_size, max_length, 1)
            y_pred = model(node_features, batch_mask, adjacency_matrix, edge_features)

            y_true = y_true.numpy().flatten()
            y_pred = y_pred.cpu().detach().numpy().flatten()
            y_mask = np.where(y_true != 0., 1, 0)

            times = 0
            for true, pred in zip(y_true, y_pred):
                if true != 0.:
                    times += 1
                    test_result['label'].append(true)
                    test_result['prediction'].append(pred)
                    test_result['loss'].append(np.abs(true - pred))
            assert times == np.sum(y_mask)
    test_result['r2'] = metrics.r2_score(test_result['label'], test_result['prediction'])
    test_result['best_valid_loss'] = checkpoint['best_valid_loss']
    return test_result


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seeds", default=np.random.randint(10000))
    parser.add_argument("--gpu", type=str, help='gpu', default=-1)
    parser.add_argument("--dataset", type=str, help='nmrshiftdb/DFT8K_DFT/DFT8K_FF/Exp5K_DFT/Exp5K_FF', default='nmrshiftdb')
    parser.add_argument("--element", type=str, help="1H/13C", default='1H')
    args = parser.parse_args()

    # load options
    model_params, train_params = get_options(args.dataset)

    # init device and seed
    print(f"Seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        train_params['device'] = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        train_params['device'] = torch.device('cpu')

    # load data
    with open(f'./Data/{args.dataset}/preprocess/graph_{args.element}_train.pickle', 'rb') as f:
        [train_all_mol, train_all_cs] = pkl.load(f)
    with open(f'./Data/{args.dataset}/preprocess/graph_{args.element}_test.pickle', 'rb') as f:
        [test_mol, test_cs] = pkl.load(f)

    print('=' * 20 + ' begin train ' + '=' * 20)

    # calculate the padding
    model_params['max_length'] = max(max([data.GetNumAtoms() for data in train_all_mol]),
                                     max([data.GetNumAtoms() for data in test_mol]))
    print(f"Max padding length is: {model_params['max_length']}")

    # split dataset
    if args.dataset == 'nmrshiftdb':
        train_mol, valid_mol, train_cs, valid_cs = train_test_split(
            train_all_mol, train_all_cs, test_size=0.05, random_state=args.seed)
    else:
        train_mol, valid_mol, train_cs, valid_cs = train_test_split(
            train_all_mol, train_all_cs, test_size=500, random_state=args.seed)

    # load dataset, data_mean=0, data_std=1 for no use
    train_dataset = construct_dataset(train_mol, train_cs, model_params['d_atom'], model_params['d_edge'],
                                      model_params['max_length'])
    valid_dataset = construct_dataset(valid_mol, valid_cs, model_params['d_atom'], model_params['d_edge'],
                                      model_params['max_length'])
    test_dataset = construct_dataset(test_mol, test_cs, model_params['d_atom'], model_params['d_edge'],
                                     model_params['max_length'])

    # calculate total warmup factor and steps
    train_params['warmup_factor'] = 0.2 if args.element == '1H' else 1.0
    train_params['total_warmup_steps'] = \
        int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
    print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

    # define a model
    model = make_model(**model_params)
    model = model.to(train_params['device'])

    # train and valid
    print(f"train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")
    best_valid_result = model_train(model, train_dataset, valid_dataset, model_params, train_params, args.dataset, args.element)
    best_valid_csv = pd.DataFrame.from_dict({'actual': best_valid_result['label'], 'predict': best_valid_result['prediction'], 'loss': best_valid_result['loss']})
    best_valid_csv.to_csv(f'./Result/{args.dataset}/best_valid_result_{args.dataset}_{args.element}.csv', sep=',', index=False, encoding='UTF-8')

    # test
    checkpoint = torch.load(f'./Model/{args.dataset}/best_model_{args.dataset}_{args.element}.pt')
    print('=' * 20 + ' summary ' + '=' * 20)
    test_result = model_test(checkpoint, test_dataset, model_params, train_params)
    print('Seed: {}, best valid loss: {:.4f}, test loss: {:.4f}, test r2: {:.4f}'
          .format(args.seed, test_result['best_valid_loss'], np.mean(test_result['loss']), test_result['r2']))
    test_csv = pd.DataFrame.from_dict({'actual': test_result['label'], 'predict': test_result['prediction']})
    test_csv.to_csv(f'./Result/{args.dataset}/best_test_result_{args.dataset}_{args.element}.csv', sep=',', index=False, encoding='UTF-8')
    print('=' * 20 + " finished!" + '=' * 20)
