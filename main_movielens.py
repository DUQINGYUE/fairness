from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal, xavier_uniform
from torch.distributions import Categorical
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import random
import argparse
import pickle
import json
import logging
import sys, os
import subprocess
from tqdm import tqdm
tqdm.monitor_interval = 0
from utils import *
from preprocess_movie_lens import *
from transD_movielens import *
import joblib
from collections import Counter, OrderedDict
import ipdb
sys.path.append('../')
import gc
from model import *
import config

# ftensor = torch.FloatTensor
ltensor = torch.LongTensor

v2np = lambda v: v.data.cpu().numpy()
USE_SPARSE_EMB = True

torch.backends.cudnn.enabled=False
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_tqdm', type=bool, default=False, help='')
    parser.add_argument('--save_dir', type=str, default='./results/MovieLens/', help="output path")
    parser.add_argument('--model_save_dir', type=str, default='./results/', help="output path")
    parser.add_argument('--do_log', action='store_true', help="whether to log to csv")
    parser.add_argument('--load_transD', action='store_true', help="Load TransD")
    parser.add_argument('--load_filters', action='store_true', help="Load TransD")
    parser.add_argument('--freeze_transD', action='store_true', help="Load TransD")
    parser.add_argument('--test_new_disc', action='store_true', help="Load TransD")
    parser.add_argument('--remove_old_run', action='store_true', help="remove old run")
    parser.add_argument('--use_cross_entropy', action='store_true', help="DemPar Discriminators Loss as CE")
    parser.add_argument('--data_dir', type=str, default='./data/', help="Contains Pickle files")
    parser.add_argument('--api_key', type=str, default=" ", help="Api key for Comet ml")
    parser.add_argument('--project_name', type=str, default=" ", help="Comet project_name")
    parser.add_argument('--workspace', type=str, default=" ", help="Comet Workspace")
    parser.add_argument('--D_steps', type=int, default=10, help='Number of D steps')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs (default: 500)')
    parser.add_argument('--num_classifier_epochs', type=int, default=100, help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=9000, help='Batch size (default: 512)')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='Batch size (default: 512)')
    parser.add_argument('--gamma', type=int, default=1, help='Tradeoff for Adversarial Penalty')
    parser.add_argument('--valid_freq', type=int, default=99, help='Validate frequency in epochs (default: 50)')
    parser.add_argument('--print_freq', type=int, default=5, help='Print frequency in epochs (default: 5)')
    parser.add_argument('--embed_dim', type=int, default=20, help='Embedding dimension (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.001)')
    parser.add_argument('--margin', type=float, default=3, help='Loss margin (default: 1)')
    parser.add_argument('--p', type=int, default=1, help='P value for p-norm (default: 1)')
    parser.add_argument('--prefetch_to_gpu', type=int, default=0, help="")
    parser.add_argument('--full_loss_penalty', type=int, default=0, help="")
    parser.add_argument('--filter_false_negs', type=int, default=1, help="filter out sampled false negatives")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--use_1M', type=bool, default=False, help='Use 1M dataset')
    parser.add_argument('--report_bias', type=bool, default=True, help='Report dataset bias')
    parser.add_argument('--use_attr', type=bool, default=False, help='Initialize all Attribute')
    parser.add_argument('--use_occ_attr', type=bool, default=False, help='Use Only Occ Attribute')
    parser.add_argument('--use_gender_attr', type=bool, default=False, help='Use Only Gender Attribute')
    parser.add_argument('--use_age_attr', type=bool, default=False, help='Use Only Age Attribute')
    parser.add_argument('--use_random_attr', type=bool, default=False, help='Use a Random Attribute')
    parser.add_argument('--use_gcmc', type=bool, default=False, help='Use a GCMC')
    parser.add_argument('--dont_train', action='store_true', help='Dont Do Train Loop')
    parser.add_argument('--debug', action='store_true', help='Stop before Train Loop')
    parser.add_argument('--sample_mask', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--use_trained_filters', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--optim_mode', type=str, default='adam', help='optimizer')
    parser.add_argument('--fairD_optim_mode', type=str, default='adam_hyp2',help='optimizer for Fairness Discriminator')
    parser.add_argument('--namestr', type=str, default='', help='additional info in output filename to help identify experiments')
    parser.add_argument("--lamda",
                        type=float,
                        default=0.001,
                        help="model regularization rate")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="training epoches")
    parser.add_argument("--top_k",
                        type=int,
                        default=10,
                        help="compute metrics@top_k")
    parser.add_argument("--factor_num",
                        type=int,
                        default=32,
                        help="predictive factors numbers in the model")
    parser.add_argument("--num_ng",
                        type=int,
                        default=1,
                        help="sample negative items for training")
    parser.add_argument("--test_num_ng",
                        type=int,
                        default=99,
                        help="sample part of negative items for testing")
    parser.add_argument("--out",
                        default=True,
                        help="save model or not")
    parser.add_argument("--gpu",
                        type=str,
                        default="0",
                        help="gpu card ID")
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    #数据集选择
    if not args.use_1M:
        args.train_ratings,args.test_ratings,args.users,args.movies = make_dataset(True)
    else:
        args.train_ratings,args.test_ratings,args.users,args.movies,args.user_num, args.item_num, args.train_mat,args.longlist,args.shortlist = make_dataset_1M_new(True,args.num_ng,args.test_num_ng)
        print(args.train_ratings)
    ''' Offset Movie ID's by # users because in TransD they share the same
    embedding Layer '''
    #args.train_ratings['movie_id'] += int(np.max(args.users['user_id'])) + 1
    #args.test_ratings['movie_id'] += int(np.max(args.users['user_id'])) + 1
    if args.use_random_attr:
        rand_attr = np.random.choice(2, len(args.users))
        args.users['rand'] = rand_attr

    args.num_users = int(np.max(args.users['user_id'])) + 1
    args.num_movies = int(np.max(args.movies['movie_id'])) + 1
    args.num_ent = args.num_users + args.num_movies
    args.num_rel = 5
    #print("!!!!!!!!!!!!!!!!")
    users = np.asarray(list(set(args.users['user_id'])))
    np.random.shuffle(users)
    if args.use_1M:
        cutoff_constant = 0.9
    else:
        cutoff_constant = 0.8
    train_cutoff_row = int(np.round(len(users)*cutoff_constant))
    args.cutoff_row = train_cutoff_row
    args.users_train = users[:train_cutoff_row]
    args.users_test = users[train_cutoff_row:]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    #print("@@@@@@@@@@@@@@@@@@")
    args.outname_base = os.path.join(args.save_dir,args.namestr+'_MovieLens_results')
    args.saved_path = os.path.join(args.save_dir,args.namestr+'_MovieLens_resultsD_final.pts')
    args.gender_filter_saved_path = args.outname_base + 'GenderFilter.pts'
    args.occupation_filter_saved_path = args.outname_base + 'OccupationFilter.pts'
    args.age_filter_saved_path = args.outname_base + 'AgeFilter.pts'
    args.random_filter_saved_path = args.outname_base + 'RandomFilter.pts'

    #print("####################")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    #print("$$$$$$$$$$$$$$$")

    ##############################################################
    return args

def main(args):
    print("数据准备完整")
    train_set = KBDataset(args.train_ratings, args.prefetch_to_gpu)
    #print(train_set)
    #test_set = KBDataset(args.test_ratings, args.prefetch_to_gpu)

    #print("jinru main hanshu")
    train_fairness_set = NodeClassification(args.users_train, args.prefetch_to_gpu)
    test_fairness_set = NodeClassification(args.users_test, args.prefetch_to_gpu)
    #print("数据准备好")
    if args.prefetch_to_gpu:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        #print("构造train_loader")
        train_loader = DataLoader(train_set,batch_size=args.batch_size, shuffle=True, num_workers=4)
        #test_loader = DataLoader(test_set, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)


    if not args.use_gcmc:
        # modelD = TransD(args.num_ent, args.num_rel, args.embed_dim,\
                # args.p).to(args.device)
        modelD = TransE(args.num_ent, args.num_rel, args.embed_dim,\
                args.p).to(args.device)
    else:
        #关系得分由双线性形式给出的解码器  图卷积矩阵补全 modelD就是评分预测的模型
        #decoder = SharedBilinearDecoder(args.num_rel,2,args.embed_dim).to(args.device)
        #modelD = SimpleGCMC(decoder,args.embed_dim,args.num_ent,args.p).to(args.device)
        #print("jinru BPR")

        modelD=BPR(args.user_num, args.item_num, args.factor_num).to(args.device)
    ''' Initialize Everything to None '''
    fairD_gender, fairD_occupation, fairD_age, fairD_random = None,None,None,None
    optimizer_fairD_gender, optimizer_fairD_occupation, \
            optimizer_fairD_age, optimizer_fairD_random = None,None,None,None
    gender_filter, occupation_filter, age_filter = None, None, None
    if args.use_attr:
        attr_data = [args.users,args.movies]
        ''' Initialize Discriminators '''
        fairD_gender = GenderDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                'gender',use_cross_entropy=args.use_cross_entropy).to(args.device)
        fairD_occupation = OccupationDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        fairD_age = AgeDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)

        ''' Initialize Optimizers '''
        if args.sample_mask:
            gender_filter = AttributeFilter(args.embed_dim,attribute='gender').to(args.device)
            occupation_filter = AttributeFilter(args.embed_dim,attribute='occupation').to(args.device)
            age_filter = AttributeFilter(args.embed_dim,attribute='age').to(args.device)
            optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
            optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
            optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)

    elif args.use_occ_attr:
        attr_data = [args.users,args.movies]
        fairD_occupation = OccupationDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
    elif args.use_gender_attr:
        attr_data = [args.users,args.movies]
        fairD_gender = GenderDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                'gender',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
    elif args.use_age_attr:
        attr_data = [args.users,args.movies]
        fairD_age = AgeDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)
    elif args.use_random_attr:
        attr_data = [args.users,args.movies]
        fairD_random = RandomDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                'random',use_cross_entropy=args.use_cross_entropy).to(args.device)
        # fairD_random = DemParDisc(args.use_1M,args.embed_dim,attr_data,\
                # attribute='random',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_random = optimizer(fairD_random.parameters(),'adam', args.lr)

    if args.load_transD:
        modelD.load(args.saved_path)

    if args.load_filters:
        gender_filter.load(args.gender_filter_saved_path)
        occupation_filter.load(args.occupation_filter_saved_path)
        age_filter.load(args.age_filter_saved_path)

    ''' Create Sets '''
    fairD_set = [fairD_gender,fairD_occupation,fairD_age,fairD_random]
    filter_set = [gender_filter,occupation_filter,age_filter,None]
    optimizer_fairD_set = [optimizer_fairD_gender, optimizer_fairD_occupation,\
            optimizer_fairD_age,optimizer_fairD_random]

    ''' Initialize CUDA if Available '''
    if args.use_cuda:
        for fairD,filter_ in zip(fairD_set,filter_set):
            if fairD is not None:
                fairD.to(args.device)
            if filter_ is not None:
                filter_.to(args.device)

    if args.use_gcmc:
        if args.sample_mask and not args.use_trained_filters:
            optimizerD = optimizer(list(modelD.parameters()) + \
                    list(gender_filter.parameters()) + \
                    list(occupation_filter.parameters()) + \
                    list(age_filter.parameters()), 'adam', args.lr)
            # optimizer_fairD_gender = optimizer(list(fairD_gender.parameters()) + \
                    # list(gender_filter.parameters()),'adam', args.lr)
        else:
            optimizerD = optim.SGD(
                modelD.parameters(), lr=args.lr, weight_decay=args.lamda)
    else:
        optimizerD = optimizer(modelD.parameters(), 'adam_sparse', args.lr)

    _cst_inds = torch.LongTensor(np.arange(args.num_ent, \
            dtype=np.int64)[:,None]).to(args.device).repeat(1, args.batch_size//2)
    _cst_s = torch.LongTensor(np.arange(args.batch_size//2)).to(args.device)
    _cst_s_nb = torch.LongTensor(np.arange(args.batch_size//2,args.batch_size)).to(args.device)
    _cst_nb = torch.LongTensor(np.arange(args.batch_size)).to(args.device)

    if args.freeze_transD:
        freeze_model(modelD)

    if args.debug:
        attr_data = [args.users,args.movies]
        ipdb.set_trace()
    '''
    #原始样本的维度
    test_data = []
    with open(config.test_negative, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()

    for i in range(101):
        print(test_data[i])
    #print(args.test_ratings[0])
    #print(len(args.test_ratings[0]))
    '''
    test_data = []
    
    #一行是用户，一个正样本，99个负样本，所以一共有101个
    for i in range(0, len(args.test_ratings)):
        u = args.test_ratings[i][0]
        test_data.append([u, args.test_ratings[i][1]])
        for j in range(2, args.test_num_ng + 2):
            test_data.append([u, args.test_ratings[i][j]])
            


    test_dataset = data_utils.BPRData(
        test_data, args.item_num, args.train_mat, 0, False)
    test_loader = DataLoader(test_dataset,
                                  batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)
    if not args.dont_train:
        for epoch in tqdm(range(1, args.num_epochs + 1)):
            # 选择特定的epoch进行测试
            '''
                        if epoch % args.valid_freq == 0 or epoch == 1:
                with torch.no_grad():
                    if args.use_gcmc:
                        print("test1")
                        # rmse,test_loss = test_gcmc(test_set,args,modelD,filter_set)

                    else:
                        print("test")
                        # l_ranks,r_ranks,avg_mr,avg_mrr,avg_h10,avg_h5 = test(test_set, args, all_hash,\
                        # modelD,subsample=20)
                        # test_nce(test_set,args,modelD,epoch,experiment)
                if args.use_attr:
                    test_gender(args, test_fairness_set, modelD, fairD_gender,  epoch, filter_set)
                    test_occupation(args, test_fairness_set, modelD, fairD_occupation,  epoch, filter_set)
                    test_age(args, test_fairness_set, modelD, fairD_age, epoch, filter_set)
                elif args.use_gender_attr:
                    # test_gender(args,test_fairness_set,modelD,fairD_gender,experiment,epoch,filter_set)
                    print("测试现在有bug，先跑训练部分")
                elif args.use_occ_attr:
                    test_occupation(args, test_fairness_set, modelD, fairD_occupation,  epoch, filter_set)
                elif args.use_age_attr:
                    test_age(args, test_fairness_set, modelD, fairD_age,  epoch, filter_set)
                elif args.use_random_attr:
                    test_random(args, test_fairness_set, modelD, fairD_random,  epoch, filter_set)
                    # test_fairness(test_fairness_set,args,modelD,experiment,\
                    # fairD_random,attribute='random',\
                    # epoch=epoch)
                if args.do_log:  # Tensorboard logging
                    if args.use_gcmc:
                        # print(float(rmse))
                        print(epoch)
                        print("@@@@@@@@@@@@@@2")
                        # experiment.log_metric("RMSE",float(rmse),step=epoch)
                        # experiment.log_metric("Test Loss",float(rmse),step=epoch)
                    # else:
                    # experiment.log_metric("Mean Rank",float(avg_mr),step=epoch)
                    # experiment.log_metric("Mean Reciprocal Rank",\
                    # float(avg_mrr),step=epoch)
                    # experiment.log_metric("Hit @10",float(avg_h10),step=epoch)
                    # experiment.log_metric("Hit @5",float(avg_h5),step=epoch)

            '''
            print("开始训练")
            #print(train_loader)
            train(train_loader, epoch, args, modelD, optimizerD,fairD_set, optimizer_fairD_set, filter_set)
            gc.collect()
            HR, NDCG = [], []
            HRlong,HRshort=[],[]
            longhit=[]
            shorthit=[]
            NDCGlong,NDCGshort=[],[]
            longrank=[]
            shortrank=[]
            print("test")
            for user, item_i, item_j in test_loader:
                user = user.cuda()
                item_i = item_i.cuda()
                item_j = item_j.cuda()  # not useful when testing
                prediction_i, prediction_j = modelD(user, item_i, item_j )
                _, indices = torch.topk(prediction_i, args.top_k)
                recommends = torch.take(item_i, indices).cpu().numpy().tolist()
                gt_item = item_i[0].item()
                #计算每个用户的平均rank
                for i in range(len(recommends)):
                    if recommends[i] in args.longlist:
                        longrank.append(i)
                    if recommends[i] in args.shortlist:
                        shortrank.append(i)





                inter1 = list(set(args.longlist) & set(recommends))
                inter2=list(set(args.shortlist) & set(recommends))
                longhit.append(len(inter1))
                shorthit.append(len(inter2))
                HR.append(evaluate.hit(gt_item, recommends))
                NDCG.append(evaluate.ndcg(gt_item, recommends))
                if gt_item in args.longlist:
                    HRlong.append(evaluate.hit(gt_item, recommends))
                    NDCGlong.append(evaluate.ndcg(gt_item, recommends))
                if gt_item in args.shortlist:
                    HRshort.append(evaluate.hit(gt_item, recommends))
                    NDCGshort.append(evaluate.ndcg(gt_item, recommends))
                #hit中的长尾分布
            print("longrank: {:.3f}\tshortrank: {:.3f}".format(np.mean(longrank), np.mean(shortrank)))
            print("long: {:.3f}\tshort: {:.3f}".format(np.mean(longhit), np.mean(shorthit)))
            print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
            print("HRlong: {:.3f}\tHRshort: {:.3f}".format(np.mean(HRlong), np.mean(HRshort)))
            print("NDCGlong: {:.3f}\tNDCGshort: {:.3f}".format(np.mean(NDCGlong), np.mean(NDCGshort)))

            if args.use_attr:
                test_gender(args, test_fairness_set, modelD, fairD_gender, epoch, filter_set)
                test_occupation(args, test_fairness_set, modelD, fairD_occupation, epoch, filter_set)
                test_age(args, test_fairness_set, modelD, fairD_age, epoch, filter_set)
            elif args.use_gender_attr:
                print("test_gender")
                test_gender(args,test_fairness_set,modelD,fairD_gender,epoch,filter_set)
            elif args.use_occ_attr:
                test_occupation(args, test_fairness_set, modelD, fairD_occupation, epoch, filter_set)
            elif args.use_age_attr:
                test_age(args, test_fairness_set, modelD, fairD_age, epoch, filter_set)
            elif args.use_random_attr:
                test_random(args, test_fairness_set, modelD, fairD_random, epoch, filter_set)
                # test_fairness(test_fairness_set,args,modelD,experiment,\
                # fairD_random,attribute='random',\
                # epoch=epoch)


        #modelD.save(args.outname_base+'D_final.pts')
        if args.use_attr or args.use_gender_attr:
            fairD_gender.save(args.outname_base+'GenderFairD_final.pts')
        if args.use_attr or args.use_occ_attr:
            fairD_occupation.save(args.outname_base+'OccupationFairD_final.pts')
        if args.use_attr or args.use_age_attr:
            fairD_age.save(args.outname_base+'AgeFairD_final.pts')
        if args.use_random_attr:
            fairD_random.save(args.outname_base+'RandomFairD_final.pts')

        if args.sample_mask:
            gender_filter.save(args.outname_base+'GenderFilter.pts')
            occupation_filter.save(args.outname_base+'OccupationFilter.pts')
            age_filter.save(args.outname_base+'AgeFilter.pts')

    constant = len(fairD_set) - fairD_set.count(None)
    '''
        if args.test_new_disc:
        if args.test_new_disc:
            args.use_attr = True
         #Training Fresh Discriminators
        args.freeze_transD = True
        attr_data = [args.users,args.movies]
        if args.use_random_attr:
            new_fairD_random = DemParDisc(args.use_1M,args.embed_dim,attr_data,\
                    attribute='random',use_cross_entropy=args.use_cross_entropy).to(args.device)
            new_optimizer_fairD_random = optimizer(new_fairD_random.parameters(),'adam', args.lr)

        freeze_model(modelD)
        with experiment.test():
            # Train Classifier 
            if args.use_gender_attr or args.use_attr:
                train_gender(args,modelD,train_fairness_set,test_fairness_set,\
                        attr_data,experiment,filter_set)
            if args.use_occ_attr or args.use_attr:
                train_occupation(args,modelD,train_fairness_set,test_fairness_set,\
                        attr_data,experiment,filter_set)
            if args.use_age_attr or args.use_attr:
                train_age(args,modelD,train_fairness_set,test_fairness_set,\
                        attr_data,experiment,filter_set)
            if args.use_random_attr:
                train_random(args,modelD,train_fairness_set,test_fairness_set,\
                        attr_data,experiment,filter_set)
                # train_fairness_classifier(train_fairness_set,args,modelD,experiment,new_fairD_random,\
                        # new_optimizer_fairD_random,epoch,filter_=None,retrain=False)

        if args.report_bias:
            gender_bias = calc_attribute_bias('Train',args,modelD,experiment,\
                    'gender',epoch,[gender_filter])
            occ_bias = calc_attribute_bias('Train',args,modelD,experiment,\
                    'occupation',epoch,[occupation_filter])
            age_bias = calc_attribute_bias('Train',args,modelD,experiment,\
                    'age',epoch,[age_filter])
            gender_bias = calc_attribute_bias('Test',args,modelD,experiment,\
                    'gender',epoch,[gender_filter])
            occ_bias = calc_attribute_bias('Test',args,modelD,experiment,\
                    'occupation',epoch,[occupation_filter])
            age_bias = calc_attribute_bias('Test',args,modelD,experiment,\
                    'age',epoch,[age_filter])
        
    '''


if __name__ == '__main__':
    main(parse_args())
