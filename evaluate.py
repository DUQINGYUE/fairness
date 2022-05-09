import numpy as np
import torch
from torch.autograd import Variable


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []
	'''
	data_itr = enumerate(test_loader)

	# print("data_itr")
	for idx, p_batch in data_itr:
		p_batch = p_batch.cuda()
		p_batch_var = Variable(p_batch)
		for user, item_i, item_j in p_batch_var:
			user = user.cuda()
			item_i = item_i.cuda()
			item_j = item_j.cuda()  # not useful when testing

			prediction_i, prediction_j = model(user, item_i, item_j)
			_, indices = torch.topk(prediction_i, top_k)
			recommends = torch.take(
				item_i, indices).cpu().numpy().tolist()

			gt_item = item_i[0].item()
			HR.append(hit(gt_item, recommends))
			NDCG.append(ndcg(gt_item, recommends))
	'''
	print("进入测试函数")
	print(test_loader)
	for user,item_i,item_j in test_loader:

		
		print(user,item_i,item_j)
		user = user.cuda()
		item_i = item_i.cuda()
		item_j = item_j.cuda() # not useful when testing

		prediction_i, prediction_j = model(user, item_i, item_j,)
		_, indices = torch.topk(prediction_i, top_k)
		recommends = torch.take(item_i, indices).cpu().numpy().tolist()

		gt_item = item_i[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))



	return np.mean(HR), np.mean(NDCG)
